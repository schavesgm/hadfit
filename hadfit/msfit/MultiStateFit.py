# -- Import some built-in dependencies
import re
from functools import lru_cache

# -- Import some third-party modules
import numpy as np
import lmfit as lm

# -- Import user-defined modules
from hadfit import Hadron
from hadfit import Model
from hadfit import bootstrap_fit
from .utils import clean_parenthesis
from .utils import compute_best_estimate
from .utils import select_initial_mass

class MultiStateFit:
    """ Class to fit hadronic data to multistate ansatz. """

    def __init__(self, hadron: Hadron, ansatz: Model, Ns_max: int, normalise: bool, prop: float = 1.0):

        # Assert the proportion of the total extent is a number between 0 and 1.
        assert 0 < prop <= 1, f'{prop_extent=} should be a number between 0 and 1'

        # Save the hadron and the ansatz in the class
        self.hadron, self.ansatz = hadron, ansatz

        # Save the maximum number of states used in the model and whether to normalise
        self.Ns_max, self.normalise = Ns_max, normalise

        # Save the proportion of the total extent and the maximum Nk used
        self.prop, self.maxNk = prop, int(prop * (self.hadron.Nk // 2 + 1))

        # Ansatz parameters to be used in the class
        ansatz_params = str(ansatz.symb_parameters).replace('(', '').replace(')', '')

        # Generate the string for the first parameter and the second
        self.spnames = tuple(s.strip() for s in ansatz_params.split(','))

        # Echo some important information to stdout
        print(f' -- Hadron used in the calculation: {self.hadron}', flush=True)
        print(f' -- Maximum number of states used: {self.Ns_max}', flush=True)
        print(f' -- Normalising: {self.normalise}', flush=True)
        print(f' -- Maximum Nk used in the calculation: {self.maxNk}', flush=True)
        print(f' -- Ansatz used: {ansatz.expr}', flush=True)
        print(f'    Note that the ansatz must contain a mass and amplitude', flush=True)
        print(f'    Ansatz parameters are: {ansatz_params}', flush=True)
        print(f'      - {self.spnames[0]} will be treated as amplitude', flush=True)
        print(f'      - {self.spnames[1]} will be treated as mass', flush=True)

        # Save the inverse of the covariance matrix to optimise the code
        self.inv_cov = np.linalg.inv(self.hadron.covariance_matrix(folded=True) / self.hadron.Nc)

        # Generate several models to fit the data in a dictionary
        self.models = {}

        # Append the models to the dictionary
        for ns in range(self.Ns_max):
            self.models[f's{ns}'] = self.__generate_model(ns + 1)

    # -- Public methods of the class {{{
    def analyse_ground_mass(self, fitres: dict, mc_iters: int = 1000) -> dict:
        """ Extract the best estimate of the ground mass from a dictionary
        of results extracted from the method extract_ground_mass. The ground
        mass is extracted using the Akaike information criterion as the
        the way to select the best models among all possible ones. 

        Due to the unstability of the fits, not all results are treated
        as valid. Results whose ground mass is too small (less than 180MeV
        for a_t^{-1} ~ 6GeV), results whose errors are not present and
        results whose chi_squared values are not valid are discarded from
        the analysis.

        The final results are extracted using the weighted average of the
        valid results. The weights are calculated using the relative 
        likelihood of data description defined as

                l(m2) = exp(- 0.5 (AICc(m2) - AICc_min)),

        where AICc_min corresponds to the lowest AICc among all models.
        
        --- Parameters
        fitres: dict[int, dict[str, dict]]
            Dictionary containing nk as keys and a dictionary of results as
            values.
        mc_iters: int
            Number of MonteCarlo iterations used to generate the best mass 
            and its sampling error. Default is 1000.
        
        --- Returns
        dict[int, dict[str, float]]
            Dictionary containing the best ground mass estimate for each nk.
        """

        # Dictionary that will contain the results
        best_estimate = {}

        # Iterate through all keys and values in the dictionary
        for nk, results in fitres.items():

            # Collect all correct M0 using some conditions
            relevant_info = []

            # Iterate for all information contained in the result dictionary
            for info in results.values():

                # Control variables to clean the data
                non_NaN_rchi  = not np.isnan(info['rchi'])
                non_None_dM0  = info['dM0'] is not None
                not_too_small = info['M0'] > 0.030015

                # If the error is not None, then check for Nan
                if non_None_dM0:
                    non_NaN_dM0 = not np.isnan(info['dM0'])

                # All booleans must be true to be feasible
                control = non_NaN_rchi and non_NaN_dM0 and non_None_dM0 and not_too_small

                if control: relevant_info.append(info)

            # Get the best model result for the given nk
            best_estimate[nk] = compute_best_estimate(np.array(relevant_info), mc_iters)
        
        return best_estimate

    @lru_cache(maxsize = 1)
    def estimate_ground_mass(self, kmin_min: int, kmin_max: int, use_bootstrap: bool) -> dict:
        """ Estimate the ground mass for all times between kmin_min and Nk // 2 - 2,
        being Nk the number of variables (times) in the correlation function. The result
        is a dictionary containing the different nk as keys and a dictionary of estimates
        of the ground mass as values.

        The ground mass is estimated for different minimum kmin starting from kmin_min to
        kmin_max. For each of this kmin, different minimum boundary values are used to
        try shrinking the region in which the parameters can lie.

        Example: kmin_min = 2, kmin_max = 3 means that first we fit for all times in
        [2, Nk // 2 - 2] and then we fit for all times between [3, Nk // 2 - 2]. The difference
        is that for the first case, the initial parameters are estimated with the largest
        window starting at kmin = 2, while the second case uses initial parameters estimated
        with the largest window starting at kmin = 3. Varying kmin reduces the bias of
        choosing kmin.

        --- Parameters
        kmin_min: int
            Minimum nk that can be used as starting point for all the other windows.
        kmin_max: int
            Maximum nk that can be used as starting point for all the other windows.
        use_bootstrap: bool
            Use bootstrap to estimate the errors in the parameters. Computationally expensive.
        
        --- Returns
            Dictionary containing the estimates for each of the nk values. For each nk there
            is another dictionary with multiple entries, each entry contains information of the
            fit.
        """

        # Assert some conditions on the minimum and maximum values
        assert 0 <= kmin_min < kmin_max,  f'{kmin_min=} should be in [0, {kmin_max}]'
        assert kmin_max < self.maxNk - 2, f'{kmin_max=} should be in ({kmin_min}, {self.maxNk-2}]'

        # Dictionary that will contain the results
        fitres = {nk: {} for nk in range(kmin_min, self.maxNk - 3)}

        # Fit the data using every kmin
        for kmin in range(kmin_min, kmin_max + 1):

            # Use different minimum values for the amplitude
            for amp_min in [-2.0, -1.0, -0.5, -0.1]:

                # Estimate the initial parameters using correlated fits
                init_C = self.estimate_initvals(kmin, True, amp_min)

                # Fit the data using the initial parameters
                fitres = self.__fit_using_initvals(fitres, init_C, kmin, use_bootstrap, f'{amp_min}C')

                # Estimate the initial parameters using uncorrelated fits
                init_U = self.estimate_initvals(kmin, False, amp_min)

                # Fit the data using the initial parameters
                fitres = self.__fit_using_initvals(fitres, init_U, kmin, use_bootstrap, f'{amp_min}U')

        return fitres

    @lru_cache(maxsize = 1)
    def estimate_initvals(self, kmin: int, use_correlated: bool, amp_min: float = -2.0, inv_ak: float = 5997) -> lm.Parameters:
        """ Estimate the initial values of all parameters involved in the model.

        If a model contains Ns states with 2 parameters each, it will estimate 2 * Ns
        initial values. The algorithm used to estimate the parameters is the following:

        1. First, initialise all the parameter values to 1.0 and set the minimum and maximum
        values to min_val and max_val respectively.
        2. Iterate for each of the models, starting with the lowest number of states. (ns)
            2.1. Isolate the correlation function data substracting the model of ns parameters
                at the parameter values contained in the variable params. See __isolate_corr
                for more information.
            2.2. Estimate the initial value of the mass for this state using the effective mass
                or 1.45 times the previous parameter. See __effective_mass for more information.
            2.3. Fit the model to the isolated correlator using the model with one state to obtain
                an estimate of the parameters for the ns-th state. The fit is done starting at the
                fit window of ns = 1 states. See generate_windows for more information.
            2.4. To enhance the estimation, do for ns > 0
                2.4.1. Fix all the parameter values for n < ns, that is, do not vary them on the
                    fit. Fit the whole correlation function to the model with ns states only varying
                    the parameters of the ns-th model. The fit is carried out using the starting window
                    defined by the method generate_windows.
                2.4.2. Fit the model with ns states to the whole correlation function varying all
                    parameters. Use the previous fitted values as starting points. The fit is carried
                    out starting from the point defined by the method generate_windows.

        --- Parameters:
        kmin: int
            Minimum value of the regressor used in the fit. Example: kmin = 3 means that
            the model with largest number of states will fit from 3 to Nk / 2, being Nk
            the number of variables (Nt) in the correlation function.
        use_correlated: bool
            Flag to use correlated fits; possibly unstable but more realistic.
        amp_min: float = -2
            Minimum value that the amplitude parameter is allowed to take. Varying this number
            could enhance the estimation of the parameters.
        max_val: float = 4
            Maximum value that each parameter is allowed to take. Varying this number could 
            enchance the estimation of the parameters.
        inv_ak: float
            Inverse of the lattice spacing in the non-integrated direction. For example, the 
            temporal lattice spacing for thermal correlation functions.

        --- Returns:
        lm.Parameters: lmfit Parameters object that contains the estimation for each parameter.
        """

        # Assert kmin is inside the bounds
        assert 0 <= kmin <= self.maxNk, f'{kmin = } must be in [0, {self.maxNk}]'

        # Construct all the minimum fit windows for all states
        kmin_wind = [self.generate_windows(kmin, ns + 1) for ns in range(self.Ns_max)]

        # Minimum and maximum values to be used in the effective mass
        t0_eff, tf_eff = int(0.65 * self.maxNk), int(0.85 * self.maxNk)

        # If use_correlated is true, then use the inv_cov
        inv_cov = self.inv_cov if use_correlated else None

        # Parameters object to be used in the calculation
        params = self.models[f's{self.Ns_max-1}'].params

        # Maximum length used in the estimation
        max_Nk = self.hadron.Nk // 2 + 1

        # Set the parameters default values and bounds
        for param in params:

            # Set the amplitude parameters
            if self.spnames[0] in str(param):

                # Set the initial parameters for the amplitude
                params[param].value = 1.0
                params[param].min   = amp_min
                params[param].max   = 5.0

            # Set the mass parameters
            elif self.spnames[1] in str(param):

                # Select the initial mass of the hadron
                init_mass = select_initial_mass(self.hadron, inv_ak)

                # Set the value and minimum and maximum limits
                params[param].value = init_mass
                params[param].max   = 1.0
                params[param].min   = max(100 / inv_ak, 0.2 * init_mass)

        # Iterate for each model, from lowest number of parameters to largest
        for ns in range(self.Ns_max):

            # Isolate the correlation function
            isol_corr = self.__isolate_corr(ns, params)

            # Set the masses to the corresponding values
            if ns < 2:
                # Get the initial value of mass to be used
                M_init = (1 + 0.50 * ns) * params[f'{self.spnames[1]}0']

                # Estimate the effective mass using the initial mass value
                mass_est = np.mean(self.__effective_mass(np.mean(isol_corr, axis = 0), t0_eff, tf_eff, M_init))

                # Change the mass depending on value
                mass_est = mass_est if mass_est > 100 / inv_ak else M_init

            else:
                mass_est = 1.50 * params[f'{self.spnames[1]}{ns - 1}']

            # Set the initial parameter to the mass estimate
            params[f'{self.spnames[1]}{ns}'].value = mass_est

            # Set the parameters in the model lowest model
            self.models['s0'].set_parameters(params)

            # Minimise the lowest model setting to extract an estimate of the parameters
            result_isol = self.__fit_model(isol_corr, 0, kmin_wind[0], max_Nk, False, inv_cov=inv_cov)

            # Set the result parameter values in the appropiate state in params
            for name in result_isol.params:
                params[name.replace('0', f'{ns}')].value = result_isol.params[name].value

            # For all models with more than one state, fix some states to obtain better estimates
            if ns > 0:

                # Set the correct parameters in the model with ns states
                self.models[f's{ns}'].set_parameters(params)

                # Fix all parameters whose state is smaller than ns
                for name in self.models[f's{ns}'].params:
                    if ns > int(re.match('\w+(\d+)', name).group(1)):
                        self.models[f's{ns}'].set_param_hint(name, vary=False)

                # Fit to the original correlator using the fixed parameters
                result_fixed = self.__fit_model(
                    self.hadron.fold_and_normalise(True, self.normalise), ns,
                    kmin_wind[ns], max_Nk, False, inv_cov=inv_cov
                )

                # Set the result parameter values in the appropiate state in params
                for name in result_fixed.params:
                    params[name].value = result_fixed.params[name].value

                # Set the new parameters in the model
                self.models[f's{ns}'].set_parameters(params)

                # Unfix all parameters in the model
                for name in self.models[f's{ns}'].params:
                    self.models[f's{ns}'].set_param_hint(name, vary=True)

                # Fit the original correlator varying all parameters
                result_unfixed = self.__fit_model(
                    self.hadron.fold_and_normalise(True, self.normalise), ns, kmin_wind[ns],
                    max_Nk, False, inv_cov=inv_cov
                )

                # Set the result parameter values in the appropiate state in params
                for name in result_fixed.params:
                    params[name].value = result_fixed.params[name].value

        # Set all parameters in the models to the default values
        for model in self.models.values():
            for param in model.params:
                model.set_param_hint(param, value=1.0, min=-np.inf, max=np.inf)

        return params

    @lru_cache(maxsize = 6)
    def generate_windows(self, kmin: int, ns: int) -> int:
        """ Calculate the starting nk value for the model with number of states provided 

        The nk value is calculated using the following function:

            FW(ns) = (w1(ns) * FW(ns+1) + w2(ns) * Nk // 2) // (w1(ns) + w2(ns))

        unless ns == Ns_max, in which FW(ns) = kmin. The algorithm is recursive.

        --- Parameters
        kmin: int
            Minimum nk value used in the fit. It corresponds to the state with largest
            number of states.
        ns: int
            Number of states in the model for which we would like to generate the window.

        --- Returns
        int
            Initial window for the model with ns number of states.
        """

        # Assert some properties in num_states
        assert ns > 0,            f'{ns = } must be at least 1'
        assert ns <= self.Ns_max, f'{ns = } must be less than {self.Ns_max = }'

        # If the number of states is the number of maximum states in the fit
        if (ns == self.Ns_max): return kmin

        # Omega table defining the function. The table can be changed
        omega_1 = {1: 2, 2: 4, 3: 6}
        omega_2 = {1: 1, 2: 1, 3: 1}

        # Transform the number of states to 3 if num_states if larger than 3
        ns = 3 if ns > 3 else ns

        # Select the correct omegas
        w1, w2 = omega_1[ns], omega_2[ns]

        # Generate the following state value -- Recursive
        next_kmin = self.generate_windows(kmin, ns + 1)

        return (w1 * next_kmin + w2 * int(self.hadron.Nk // 2)) // (w1 + w2)
    # -- }}}

    # # -- Private methods of the class {{{
    def __double_estimate_params(
        self, iparams: lm.Parameters, pparams: list, ns: int, nk: int, max_Nk: int, use_bootstrap: bool, inv_cov: np.ndarray
        ) -> list:
        """ Fit a model at a given nk two times: first, using iparams as initial parameters;
        second, using the parameters obtained in the previous nk as initial parameters. Returns
        a list of two MinimizerResults.

        --- Parameters
        iparams: lm.Parameters
            Initial parameters to be used in the fit. Inmutable.
        pparams: list[lm.Parameters]
            List containing the initial parameters coming from the fit at the previous time for all
            models. Through ns, the function will select the appropiate one.
        ns: int
            Integer that selects the model containing ns + 1 states.
        nk: int
            Initial variable (time) used in the fit.
        max_Nk: int
            Maximum variable (time) used in the fit.
        use_bootstrap: bool
            Use bootstrap to estimate the standard errors. Computationally expensive.
        inv_cov: np.ndarray
            Inverse of the covariance matrix used to perform correlated fits.

        --- Returns
        list[lm.model.MinimizerResult]
            List containing the results for each of the two (possible) minimisation tasks.
        """

        # Set the initial parameters in the model
        self.models[f's{ns}'].set_parameters(iparams)

        # Append the result using the initial_parameters as initial values
        res_list = [
            self.__fit_model(
                self.hadron.fold_and_normalise(True, self.normalise), ns, nk,
                max_Nk, use_bootstrap, inv_cov=inv_cov
            )
        ]

        # If previous parameters are found, then use them
        if pparams[ns] is not None:

            # Set the initial parameters in the model
            self.models[f's{ns}'].set_parameters(pparams[ns])

            # Append the result using the initial_parameters as initial values
            res_list.append(
                self.__fit_model(
                    self.hadron.fold_and_normalise(True, self.normalise), ns, nk,
                    max_Nk, use_bootstrap, inv_cov=inv_cov
                )
            )

        # Set the previous parameters to the new results
        pparams[ns] = res_list[0].params

        return res_list

    def __clean_result(self, min_result: lm.model.ModelResult, ns: int):
        """ Extract the relevant information from a ModelResult for the model
        with ns states. The information contains the estimate of the ground
        mass and its error, as well as the corrected Akaike Information Criterion
        and the reduced chi-squared value at the minimum.

        --- Parameters
        min_result: lm.model.ModelResult
            Minimiser result of a model with ns states. Contains all the meaninful
            information about the fit.
        ns: int
            Integer that selects the model containing ns + 1 states.

        --- Returns
        dict:
            Dictionary containing the most relevant information of the fit. Contains:
            - M0: [str, float]
                Estimate of the ground mass contained in min_result
            - dM0: [str, float]
                Estimate of the standard error of M0 contained in min_result
            - AICc: [str, float]
                Estimate of the corrected Akaike Information Criterion
            - rchi: [str, float]
                Estimate of the chi-squared value.
        """

        # Number of points in the fit and parameters used
        n, k = min_result.ndata, min_result.nvarys

        # List of possible states in the model used
        order = [s for s in range(ns + 1)]

        # Order the states to obtain the ground state
        for s0 in range(ns):

            # Retrieve the values for the mass and amplitude
            Ms0 = min_result.params[f'{self.spnames[1]}{s0}'].value
            As0 = min_result.params[f'{self.spnames[0]}{s0}'].value

            for sc in range(s0, ns):
                
                # Retrieve the values for the comparing states
                Asc = min_result.params[f'{self.spnames[0]}{sc}'].value

                # Conditions used to swap states
                cond_A = As0 < Asc
                cond_B = Ms0 < 0.025 or Ms0 > 0.8
                cond_C = As0 < 0.0 and Asc > 1

                # If any condition is met, then swap orders
                if cond_A or cond_B or cond_C:
                    order[s0], order[sc] = order[sc], order[s0]

        # Obtain the error
        dM0 = min_result.params[f'{self.spnames[1]}{order[0]}'].stderr

        # If the error is not None, try multiplying by redchi
        if dM0 is not None:
            dM0 *= min_result.redchi if min_result.redchi < 20 else 1.00

        # Return a dictionary with useful information
        return {
            'M0':   min_result.params[f'{self.spnames[1]}{order[0]}'].value,
            'dM0':  dM0,
            'AICc': min_result.aic + (2 * k ** 2 + 2 * k) / (n - k - 1),
            'rchi': min_result.redchi,
        }

    def __fit_using_initvals(
        self, fitres: dict, params: lm.Parameters, kmin: int, use_bootstrap: bool, prefix: str
        ) -> dict:
        """ Fit all models available for all times between kmin and Nk // 2 - 2 using
        the initial parameters in params. The errors can be estimated using bootstrap.

        The fit results at a given nk are stored at the nkth entry in the fitres dictionary.
        
        --- Parameters
        fitres: dict
            Dictionary where the resulting parameters will be stored.
        params: lm.Parameters
            Initial parameters used to estimate the models.
        kmin: int
            Minimum nk that defines the fit window.
        use_bootstrap: bool
            Use bootstrap to estimate the standard errors of the parameters. Expensive.
        prefix: str
            Prefix used to localise the current fit in the dictionary of results

        --- Returns
        dict:
            Dictionary containing the results of the fit
        """

        # Set the initial parameters for all models
        for model in self.models.values():
            model.set_parameters(params)

        # Previous parameters for all states
        prev_params = [None] * self.Ns_max

        # Iterate through all times in the fit
        for nk in range(kmin, self.maxNk - 2):

            # Fit to the different number of states at different times
            for ns in range(self.Ns_max):

                # Only fit when there are several degrees of freedom available
                if (self.maxNk - nk) > 2 * (ns + 1) + 1:

                    # Estimate twice the number of parameters if possible
                    fit_results = self.__double_estimate_params(
                        params, prev_params, ns, nk, self.maxNk, use_bootstrap, self.inv_cov
                    )

                    # Extract the meaningful information for each result
                    for i, s in enumerate(fit_results):
                        fitres[nk][f'{prefix}{kmin}s{ns}{i}'] = self.__clean_result(s, ns)

        return fitres

    def __fit_model(
        self, data: np.ndarray, ns: int, kmin: int, kmax: int, use_bootstrap: bool, inv_cov: np.ndarray = None, **kwargs
    ) -> lm.model.ModelResult:
        """ Fit the model with ns states to a given dataset. The fit can be done using the
        correlated maximum likelihood estimate or not depending on inv_cov. The standard
        errors can be estimated using boostrap with increased computational cost.

        In the case in which the correlation function data contains Nk variables, the value
        of kmin determines from which starting variable the fit is carried out. Example: if
        kmin = 5, the fit is carried out from using the correlation function for [kmin, Nk // 2].
        The correlation function data is always folded.

        --- Parameters
        data: np.ndarray
            Correlation function data used to fit the data. Should be a 2-dimensional array
        ns: int
            Select the state with ns number of states
        kmin: int
            Initial variable (time) to be used in the fit. The fit is carried out from kmin to
            the end of the dataset.
        use_bootstrap: bool
            Use bootstrap to estimate the standard errors. Computationally expensive.
        inv_cov: np.ndarray = None
            Inverse of the covariance matrix of the data. If passed, the fit is carried out using
            the correlated maximum likelihood estimate.
        **kwargs
            Other parameters to be passed to the fitting routine.

        --- Returns
        lm.model.ModelResult
            Result of the fitting procedure.
        """
            
        # If inv_cov is not None, then crop it
        inv_cov = inv_cov[kmin:kmax, kmin:kmax] if inv_cov is not None else None

        # Obtain the regressor string to pass the nk accordingly
        regr_str = clean_parenthesis(self.ansatz.symb_regressors).replace(',', '')

        if not use_bootstrap:
            return self.models[f's{ns}'].fit(
                np.mean(data, axis=0)[kmin:kmax], inv_cov=inv_cov,
                **{regr_str: self.hadron.nk(folded=True)[kmin:kmax]}, **kwargs
            )
        else:
            return bootstrap_fit(
                self.models[f's{ns}'], self.models[f's{ns}'].params,
                data[:,kmin:kmax], **{regr_str: self.hadron.nk(folded=True)[kmin:kmax]},
                **kwargs
            )

    def __isolate_corr(self, ns: int, params: lm.Parameters) -> np.ndarray:
        """ Method used to isolate ns states from the correlation function data.

        The method computes the difference between the correlation function data
        (raw data of (Nc, Nk) variables) and the model of ns states evaluated
        at the parameters provided. The method implements

        I(nk) = self.hadron.data - (ns != 0) * self.model[ns](params) 

        --- Parameters
        ns: int
            Number of states to be eliminated from the correlation function data.
        params: lm.Parameters
            Parameters to be passed to the model

        --- Returns
        np.ndarray
            Dataset of Nc samples of Nk variables containing I(nk)
        """

        # Function to filter the parameter in the model
        def filter_f(d: tuple):
            """ Get all parameters whose state is lower or equal to ns """
            return ns >= int(re.match('\w+(\d+)', d[0]).group(1))

        # Clean the parameters to only use the useful ones for ns
        ns_params = dict(filter(filter_f, params.items()))

        # Fold and normalise the data
        data = self.hadron.fold_and_normalise(True, self.normalise)

        # Return the isolated correlation function
        return np.abs(data - (ns != 0) * self.models[f's{ns}'](self.hadron.nk(True), **ns_params))

    def __effective_mass(self, corr: np.ndarray, k0_eff: int, kf_eff: int, M_init: float) -> np.ndarray:
        """ Compute the effective mass using some data. The effective mass is
        calculated for several times between t0_eff and tf_eff. The effective
        mass is calculated using the mass that solves

                ansatz(nk, M) / ansatz(nk+1, M)) - corr(nk) / corr(nk+1) = 0

        --- Parameters
        corr: np.ndarray
            Correlation function used to define the function above.
        k0_eff: int 
            Minimum variable (time) for which we would like to solve the function above
        kf_eff: int
            Maximum variable (time) for which we would like to solve the function above
        M_init: float
            Initial value for the mass used in the solver.

        --- Returns
        np.ndarray
            Effective mass extracted for all variables inside [k0_eff, kf_eff]. The first
            value corresponds to the effective mass extracted solving the equation above
            for nk = k0_eff. The last value corresponds to nk = kf_eff.
        """

        # Import the needed function
        from scipy.optimize import fsolve

        # Assert some conditions on the parameters
        assert k0_eff >= 0, f'{k0_eff = } must be positive'
        assert k0_eff < kf_eff <= self.maxNk, f'{kf_eff=} must hold {k0_eff} < kf_eff <= {self.maxNk}'

        # Obtain the function for the lowest model to be used in the effective mass
        ansatz = self.models['s0'].function

        # Function used to solve the equation
        def to_solve(M: float, nk: int):
            # Generate the dictionary to be passed to the ansatz function
            arg_dict = {
                f'{self.spnames[0]}0': 1.0, f'{self.spnames[1]}0': M
            }

            # Return the function whose roots should be found
            return ansatz(nk, **arg_dict) / ansatz(nk+1, **arg_dict) - corr[nk] / corr[nk+1]

        # List containing all effective masses
        eff_masses = []

        # Iterate to compute the effective masses
        for nk in range(k0_eff, kf_eff):
            eff_masses.append(fsolve(to_solve, M_init, args=(nk))[0])

        return np.array(eff_masses)

    def __generate_model(self, Ns: int) -> Model:
        """ Generate a model adding ns times the ansatz provided in the fit.

        --- Parameters:
        Ns: int
            Add the model Ns times.

        --- Returns:
        Model:
            Model composed by Ns sums of self.__ansatz.
        """

        # Get the regressors used in the ansatz as a str
        regr_str = clean_parenthesis(self.ansatz.symb_regressors)

        # String representing the ansatz and parameters
        expr, params = '', []

        # Concatenate all the possible parameters
        for ns in range(Ns):

            # Add the ansatz expression to the model
            model = str(self.ansatz.expr)

            # Iterate for each parameter to substitute the string
            for param in self.ansatz.params:

                # Replace the parameter with the correct state number
                model = model.replace(f'{param}', f'{param}{ns}', 1)

                # Add the parameters to the list of parameters
                params.append(f'{param}{ns}')

            # Add the state to the whole multistate model
            expr += model  + ' + ' if ns != Ns - 1 else model

        # Join the params into a unique string
        params = ','.join(params)

        # Return the correct model
        return Model(expr, regr_str, params)
    # -- }}}

    # -- Magic methods of the class {{{
    def __str__(self) -> str:
        return f'<MultiStateFit: {self.ansatz}, {self.hadron}: Ns = {self.Ns_max}>'

    def __repr__(self) -> str:
        return f'<MultiStateFit: {self.ansatz}, {self.hadron}: Ns = {self.Ns_max}>'
    # -- }}}

if __name__ == '__main__':
    pass
