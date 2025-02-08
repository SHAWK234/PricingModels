import numpy as np

class OptionPricer:
    def __init__(self, K, SZero, r, vol, M, N, T):
        self.K = K
        self.SZero = SZero
        self.r = r
        self.vol = vol
        self.M = M
        self.N = N
        self.T = T/365
        self.deltaS = 1
        self.deltaT = 1/365
        self.deltaV = 0.01
        self.dt = self.T/self.N
        self.nudt = (self.r - 0.5 * self.vol**2) * self.dt
        self.volsdt = self.vol * np.sqrt(self.dt)
        self.nudt_deltaV = (self.r - 0.5 * (self.vol + self.deltaV)**2) * self.dt
        self.volsdt_deltaV = ((self.vol + self.deltaV) * np.sqrt(self.dt))

    def calculate(self):
        payoffs  = np.zeros(self.M)
        payoffs_delta = np.zeros(self.M)
        payoffs_gamma = np.zeros(self.M)
        payoffs = np.zeros(self.M)
        final_prices = np.zeros(self.M)

        payoffs_time_decrement = np.zeros(self.M)
        payoffs_vega = np.zeros(self.M)

        for i in range(self.M):
            lnSt = np.log(self.SZero)
            lnSt_delta = np.log(self.SZero + self.deltaS)
            lnSt_gamma = np.log(self.SZero - self.deltaS)
            lnSt_time_decrement = np.log(self.SZero) 
            lnSt_vega = np.log(self.SZero)
            for j in range(self.N):
                rand_val = np.random.normal()
                rnd = self.nudt + self.volsdt * rand_val
                lnSt += rnd
                lnSt_delta += rnd
                lnSt_gamma += rnd
                lnSt_vega += (self.r - 0.5 * (self.vol + self.deltaV)**2) * self.dt + (self.vol + self.deltaV) * np.sqrt(self.dt) * rand_val
                if j < self.N - (self.deltaT*365):
                    lnSt_time_decrement += rnd

            ST = np.exp(lnSt)
            ST_delta = np.exp(lnSt_delta)
            ST_gamma = np.exp(lnSt_gamma)
            ST_time_decrement = np.exp(lnSt_time_decrement)
            ST_vega = np.exp(lnSt_vega)
            payoffs[i] = max(0, ST - self.K)
            payoffs_delta[i] = max(0, ST_delta - self.K)
            payoffs_gamma[i] = max(0, ST_gamma - self.K)
            payoffs_time_decrement[i] = max(0, ST_time_decrement - self.K)
            payoffs_vega[i] = max(0, ST_vega - self.K)

        discounted_payoff = np.exp(-self.r * self.T) * payoffs
        discounted_payoff_delta = np.exp(-self.r * self.T) * payoffs_delta
        discounted_payoff_gamma = np.exp(-self.r * self.T) * payoffs_gamma
        discounted_payoff_time_decrement = np.exp(-self.r * (self.T - self.deltaT)) * payoffs_time_decrement
        discounted_payoff_vega = np.exp(-self.r * self.T) * payoffs_vega

        option_price = discounted_payoff.mean()
        option_price_delta = discounted_payoff_delta.mean()
        option_price_gamma = discounted_payoff_gamma.mean()
        option_price_time_decrement = discounted_payoff_time_decrement.mean()
        option_price_vega = discounted_payoff_vega.mean()

        delta = (option_price_delta - option_price) / self.deltaS
        gamma = (option_price_delta - 2*option_price + option_price_gamma)/(self.deltaS**2)
        theta = -(option_price_time_decrement - option_price) 
        vega = (option_price_vega - option_price)/1
        theta = self.r * option_price - 0.5*gamma - self.r*delta*self.SZero 
        std_error = np.std(discounted_payoff) / np.sqrt(self.M)

        return option_price, delta, gamma, theta, vega



class OptionPricerPut:
    def __init__(self, K, SZero, r, vol, M, N, T):
        self.K = K
        self.SZero = SZero
        self.r = r
        self.vol = vol
        self.M = M
        self.N = N
        self.T = T/365
        self.deltaS = 1
        self.deltaT = 1/365
        self.deltaV = 0.01
        self.dt = self.T/self.N
        self.nudt = (self.r - 0.5 * self.vol**2) * self.dt
        self.volsdt = self.vol * np.sqrt(self.dt)
        self.nudt_deltaV = (self.r - 0.5 * (self.vol + self.deltaV)**2) * self.dt
        self.volsdt_deltaV = ((self.vol + self.deltaV) * np.sqrt(self.dt))

    def calculate(self):
        payoffs  = np.zeros(self.M)
        payoffs_delta = np.zeros(self.M)
        payoffs_gamma = np.zeros(self.M)
        payoffs = np.zeros(self.M)
        final_prices = np.zeros(self.M)

        payoffs_time_decrement = np.zeros(self.M)
        payoffs_vega = np.zeros(self.M)

        for i in range(self.M):
            lnSt = np.log(self.SZero)
            lnSt_delta = np.log(self.SZero + self.deltaS)
            lnSt_gamma = np.log(self.SZero - self.deltaS)
            lnSt_time_decrement = np.log(self.SZero) 
            lnSt_vega = np.log(self.SZero)
            for j in range(self.N):
                rand_val = np.random.normal()
                rnd = self.nudt + self.volsdt * rand_val
                lnSt += rnd
                lnSt_delta += rnd
                lnSt_gamma += rnd
                lnSt_vega += (self.r - 0.5 * (self.vol + self.deltaV)**2) * self.dt + (self.vol + self.deltaV) * np.sqrt(self.dt) * rand_val
                if j < self.N - (self.deltaT*365):
                    lnSt_time_decrement += rnd

            ST = np.exp(lnSt)
            ST_delta = np.exp(lnSt_delta)
            ST_gamma = np.exp(lnSt_gamma)
            ST_time_decrement = np.exp(lnSt_time_decrement)
            ST_vega = np.exp(lnSt_vega)
            payoffs[i] = max(0, self.K - ST)
            payoffs_delta[i] = max(0, self.K - ST_delta)
            payoffs_gamma[i] = max(0, self.K - ST_gamma)
            payoffs_time_decrement[i] = max(0, self.K - ST_time_decrement)
            payoffs_vega[i] = max(0, self.K - ST_vega)

        discounted_payoff = np.exp(-self.r * self.T) * payoffs
        discounted_payoff_delta = np.exp(-self.r * self.T) * payoffs_delta
        discounted_payoff_gamma = np.exp(-self.r * self.T) * payoffs_gamma
        discounted_payoff_time_decrement = np.exp(-self.r * (self.T - self.deltaT)) * payoffs_time_decrement
        discounted_payoff_vega = np.exp(-self.r * self.T) * payoffs_vega

        option_price = discounted_payoff.mean()
        option_price_delta = discounted_payoff_delta.mean()
        option_price_gamma = discounted_payoff_gamma.mean()
        option_price_time_decrement = discounted_payoff_time_decrement.mean()
        option_price_vega = discounted_payoff_vega.mean()

        delta = (option_price_delta - option_price) / self.deltaS
        gamma = (option_price_delta - 2*option_price + option_price_gamma)/(self.deltaS**2)
        theta = -(option_price_time_decrement - option_price) 
        vega = (option_price_vega - option_price)/1
        theta = self.r * option_price - 0.5*gamma - self.r*delta*self.SZero 
        std_error = np.std(discounted_payoff) / np.sqrt(self.M)

        return option_price, delta, gamma, theta, vega

pricer = OptionPricerPut(K=237.31, SZero=237.5, r=0.0418, vol=0.4696, M=1000000, N=252, T=16)
option_price, delta, gamma, theta, vega = pricer.calculate()
print(f"Option price: ${option_price:.2f}")
print(f"Delta: {delta}, Gamma: {gamma}, Theta: {theta}, Vega: {vega}")


# # Option parameters
# K = 120      # Strike price
# SZero = 100    # Initial stock price
# r = 0.05        # Annual risk-free rate
# vol = 0.25      # Annual volatility
# M = 1000000        # Number of simulations
# N = 252           # Number of time steps (typical number for an entire year)
# T = 30 / 365      # Time to expiration in years
# deltaS = 1
# deltaT = 1/365
# deltaV = 0.01
# dt = T / N

# nudt = (r - 0.5 * vol**2) * dt
# volsdt = vol * np.sqrt(dt)
# nudt_deltaV = (r - 0.5 * (vol + deltaV)**2) * dt
# volsdt_deltaV = ((vol + deltaV) * np.sqrt(dt))

# payoffs_delta = np.zeros(M)
# payoffs_gamma = np.zeros(M)
# payoffs = np.zeros(M)
# final_prices = np.zeros(M)

# payoffs_time_decrement = np.zeros(M)
# payoffs_vega = np.zeros(M)



# for i in range(M):
#     lnSt = np.log(SZero)
#     lnSt_delta = np.log(SZero + deltaS)
#     lnSt_gamma = np.log(SZero - deltaS)
#     lnSt_time_decrement = np.log(SZero) 
#     lnSt_vega = np.log(SZero)
#     for j in range(N):
#         rand_val = np.random.normal()
#         rnd = nudt + volsdt * rand_val
#         lnSt += rnd
#         lnSt_delta += rnd
#         lnSt_gamma += rnd
#         lnSt_vega += (r - 0.5 * (vol + deltaV)**2) * dt + (vol + deltaV) * np.sqrt(dt) * rand_val
#         if j < N - (deltaT*365):
#             lnSt_time_decrement += rnd

#     ST = np.exp(lnSt)
#     ST_delta = np.exp(lnSt_delta)
#     ST_gamma = np.exp(lnSt_gamma)
#     ST_time_decrement = np.exp(lnSt_time_decrement)
#     ST_vega = np.exp(lnSt_vega)
#     payoffs[i] = max(0, ST - K)
#     payoffs_delta[i] = max(0, ST_delta - K)
#     payoffs_gamma[i] = max(0, ST_gamma - K)
#     payoffs_time_decrement[i] = max(0, ST_time_decrement - K)
#     payoffs_vega[i] = max(0, ST_vega - K)

# discounted_payoff = np.exp(-r * T) * payoffs
# discounted_payoff_delta = np.exp(-r * T) * payoffs_delta
# discounted_payoff_gamma = np.exp(-r * T) * payoffs_gamma
# discounted_payoff_time_decrement = np.exp(-r * (T - deltaT)) * payoffs_time_decrement
# discounted_payoff_vega = np.exp(-r * T) * payoffs_vega

# option_price = discounted_payoff.mean()
# option_price_delta = discounted_payoff_delta.mean()
# option_price_gamma = discounted_payoff_gamma.mean()
# option_price_time_decrement = discounted_payoff_time_decrement.mean()
# option_price_vega = discounted_payoff_vega.mean()




# delta = (option_price_delta - option_price) / deltaS
# gamma = (option_price_delta - 2*option_price + option_price_gamma)/(deltaS**2)
# theta = -(option_price_time_decrement - option_price) 
# vega = (option_price_vega - option_price)/1
# theta = r * option_price - 0.5*gamma - r*delta*SZero 
# std_error = np.std(discounted_payoff) / np.sqrt(M)
# print("Vega: ", vega)
# print("Theta: ", theta)
# print("Gamma: ", gamma)
# print("Delta: ", delta)
# print("Daily Rate:", r / 365)
# print("Daily Volatility:", vol / np.sqrt(365))
# print("Discount Factor:", np.exp(-r * T))
# print("Average Final Stock Price:", np.mean(final_prices))
# print("Option price: ${:.2f} ± ${:.2f} (SE)".format(option_price, std_error))
