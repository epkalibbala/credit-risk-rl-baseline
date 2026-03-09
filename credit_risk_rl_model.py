import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)

############################################################
# 1. Simulate Credit Data
############################################################

def simulate_credit_data(n=5000):

    income = np.random.normal(5000,1500,n)
    loan_amount = np.random.normal(15000,4000,n)
    credit_score = np.random.normal(650,50,n)
    age = np.random.randint(21,65,n)

    risk_score = (
        0.0002*loan_amount
        -0.00015*income
        -0.002*credit_score
    )

    pd_prob = 1/(1+np.exp(-risk_score))

    default = np.random.binomial(1,pd_prob)

    data = pd.DataFrame({
        "income":income,
        "loan_amount":loan_amount,
        "credit_score":credit_score,
        "age":age,
        "default":default
    })

    return data


############################################################
# 2. Train PD Model (Logistic Regression)
############################################################

def train_pd_model(data):

    X = data[["income","loan_amount","credit_score","age"]]
    y = data["default"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)

    return model


############################################################
# 3. Credit Risk Environment
############################################################

class CreditEnvironment:

    def __init__(self,pd_model):

        self.pd_model = pd_model

        self.reset()

    def reset(self):

        self.credit_score = np.random.normal(650,50)
        self.income = np.random.normal(5000,1000)
        self.loan_balance = np.random.normal(15000,3000)
        self.market_rate = 0.05

        state = np.array([
            self.credit_score,
            self.income,
            self.loan_balance,
            self.market_rate
        ])

        return state


    def step(self,action):

        exposure_levels = [0,0.5,1]
        exposure = exposure_levels[action]

        borrower = np.array([
            self.income,
            self.loan_balance,
            self.credit_score,
            40
        ]).reshape(1,-1)

        pd = self.pd_model.predict_proba(borrower)[0][1]

        interest_rate = 0.15
        loss = 1

        reward = exposure*(interest_rate*(1-pd)-pd*loss)

        # state transition (dynamic system)

        self.credit_score += np.random.normal(0,5)
        self.loan_balance *= (1+self.market_rate/12)
        self.market_rate += np.random.normal(0,0.002)

        next_state = np.array([
            self.credit_score,
            self.income,
            self.loan_balance,
            self.market_rate
        ])

        done = False

        return next_state,reward,done


############################################################
# 4. Q-Learning Agent
############################################################

class QLearningAgent:

    def __init__(self):

        self.q_table = {}

        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1


    def get_state_key(self,state):

        return tuple(np.round(state,1))


    def choose_action(self,state):

        key = self.get_state_key(state)

        if np.random.rand() < self.epsilon:
            return np.random.randint(3)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(3)

        return np.argmax(self.q_table[key])


    def update(self,state,action,reward,next_state):

        s = self.get_state_key(state)
        ns = self.get_state_key(next_state)

        if s not in self.q_table:
            self.q_table[s] = np.zeros(3)

        if ns not in self.q_table:
            self.q_table[ns] = np.zeros(3)

        best_next = np.max(self.q_table[ns])

        self.q_table[s][action] += self.alpha*(
            reward
            + self.gamma*best_next
            - self.q_table[s][action]
        )


############################################################
# 5. Training Loop
############################################################

def train_agent(env,agent,episodes=2000):

    for ep in range(episodes):

        state = env.reset()

        for t in range(50):

            action = agent.choose_action(state)

            next_state,reward,done = env.step(action)

            agent.update(state,action,reward,next_state)

            state = next_state


############################################################
# 6. Main Execution
############################################################

if __name__ == "__main__":

    data = simulate_credit_data()

    pd_model = train_pd_model(data)

    env = CreditEnvironment(pd_model)

    agent = QLearningAgent()

    train_agent(env,agent)

    print("Training completed.")
