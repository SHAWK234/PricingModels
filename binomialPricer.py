import numpy as np



SZero = 238.83
K = 240
T = 15/365
r = 0.0419
N = 20
vol = 0.4585
dt = T/N
u = np.exp(vol*np.sqrt(dt))
d = 1/u
q = ((np.exp(r * dt) - d)/(u - d))

# Underlying Price over time in last tree branch
class OptionPricer:
    def __init__(self, K, SZero, r, vol, N, T):
        self.K = K
        self.SZero = SZero
        self.r = r
        self.vol = vol
        self.N = N
        self.T = T/365
        self.dt = self.T/self.N
        self.u = np.exp(self.vol*np.sqrt(self.dt))
        self.d = 1/self.u
        self.q = ((np.exp(self.r * self.dt) - self.d)/(self.u - self.d))
    
    def calculate(self):
        S = [[self.SZero]]
        for i in range(self.N):
            bus = []
            for j in range(2**i):
                bus.append(S[i][j]*self.u)
                bus.append(S[i][j]*self.d)
            S.append(bus)

        for i in range(self.N + 1):
            for j in range(2**i):
                S[i][j] = max(0, S[i][j] - self.K)

        first_index = len(S) - 2
        discount = np.exp(-self.r*self.dt)
        for i in range(self.N):
            thing = 1
            for j in range(2**first_index):
                S[first_index][j] = discount*(self.q*S[first_index + 1][thing - 1] + (1 - self.q)*S[first_index + 1][thing])
                thing += 2
            first_index -= 1

        optionPrice = S[0][0]
        return optionPrice

def findDelta(K, SZero, r, vol, N, T):
    pricer1 = OptionPricer(K=K, SZero=SZero, r=r, vol=vol, N=N, T=T)
    option_price1 = pricer1.calculate()
    pricer2 = OptionPricer(K=K, SZero=SZero + 1, r=r, vol=vol, N=N, T=T)
    option_price2 = pricer2.calculate()
    return option_price2 - option_price1


pricer = OptionPricer(K=240, SZero=238.83, r=0.0419, vol=0.4585, N=20, T=100)
option_price = pricer.calculate()
print(f"Option price: ${option_price:.2f}")

# print(f"Delta: {delta}, Gamma: {gamma}, Theta: {theta}, Vega: {vega}")


# S = [[SZero]]

# for i in range(N):
#     bus = []
#     for j in range(2**i):
#         bus.append(S[i][j]*u)
#         bus.append(S[i][j]*d)
#     S.append(bus)

# for i in range(N + 1):
#     for j in range(2**i):
#         S[i][j] = max(0, S[i][j] - K)

# first_index = len(S) - 2
# discount = np.exp(-r*dt)
# for i in range(N):
#     thing = 1
#     for j in range(2**first_index):
#         S[first_index][j] = discount*(q*S[first_index + 1][thing - 1] + (1 - q)*S[first_index + 1][thing])
#         thing += 2
#     first_index -= 1


# print(S[0][0])


