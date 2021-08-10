import gym
from gym import spaces
import pandas as pd
import numpy as np
import os
import csv
import random


# Reset all these values to some more meaningful values !
TRANSACTION_COST = 0.01  # p/th - confirmed transaction cost, bid-ask spread
DEBUG = 0  # turn on DEBUG mode or not
MU = 1  # reward multiple factor


class MarketSimulatorEnv(gym.Env):
    """
        A market simulator ecd ../.nvironment based on OpenAI gym
    """
    class_names = ["Sell", "Do nothing", "Buy"]

    price_info_cols = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    account_info_cols = [
        "realized_pnl",
        "unrealized_pnl",
        "max_net_worth",
        "volume_held",
        "cost_basis",
        "daily_pnl",
        "prev_action",
    ]

    technical_cols = [
        'HmL',
        'LmO',
        'HmO',
        'CmCtm1',
        'adj_close',
        'adj_1mr',
        'adj_2mr',
        'adj_3mr',
        'adj_12mr',
        'macd',
    ]

    def __init__(self, lookback_window_size: int = 5, transaction_cost=TRANSACTION_COST,
                 max_num_volume: int = 1000, file_name: str = None, prefix: str = "results", seed:int=2020, vectorize = False
                , noise : float = 0.01):

        """
        :param df: pd.DataFrame of OHLCV data with fundamental features into account
        :param lookback_window_size: number of past time-steps (days) considered for taking an action
        :param max_num_volume: maximum number of volume (# contracts) to take during the entire trading period
        :param file_name: save the simulation result to a csv file
        :param prefix: prefix folder to save simulated results
        """
        metadata = {'render.modes': ['live', 'file', 'none']}

        #load dataset
        df=pd.read_csv('technical_and_fundamentals1317.csv')
        
        # make directory of the file_name
        self.file_name = file_name
        self.prefix = prefix

        # set random seed
        self.seed = seed
        random.seed(self.seed)

        # save the simulation result into a directory called prefix
        if self.prefix is not None:
            os.makedirs(prefix, exist_ok=True)

        super(MarketSimulatorEnv, self).__init__()
        self.df = df
        self.nT, self.p = df.shape  # number of candlesticks in the data
        self.p = 5
        self.lookback_window_size = lookback_window_size
        self.transaction_cost = transaction_cost
        self.current_step = self.lookback_window_size  # reset current_step
        self.historical_daily_pnl = []
        self.visualization = None
        self.vectorize = vectorize

        # set max number of contracts or shares agent can buy or sell
        # the max_num_volume depends the margin an agent has for trading futures contract
        # an agent is confronted to hold position from [-max_num_volume, max_num_volume]
        self.max_num_volume = max_num_volume

        # book keeping the trading environment
        self.cost_basis = 0.   # averaged cost for a share or contract
        self.total_volume_transacted = 0.  # total number of shares sold
        self.max_net_worth = 0  # reset maximum net worth
        self.realized_pnl = 0.
        self.unrealized_pnl = 0.
        self.m2m = 0.
        self.yesterday_m2m = 0.
        self.daily_pnl = 0.
        self.sharpe_ratio = 0.
        self.daily_position = 0.
        self.daily_value = 0.
        self.account_info = None

        self.volume_held = 0       # number of shares / contracts held
        self.prev_volume_held = 0  # number of shares / contracts previously held

        self.prev_action = 0       # start the agent with a neural position

        self.reward = 0  # immediate reward
        self.done = False  # check if simulation terminates or not
        self.noise = noise
        # store trajectory of daily pnl
        self.historical_daily_pnl = []

        # action space (trading position from 100% shorting to 100% long)
        self.action_space = spaces.Discrete(3)

        # observation space (a dictionary of both price and account numpy array)
        if self.vectorize == False:
            self.observation_space = spaces.Dict({
                "price": spaces.Box(low=-np.inf, high=np.inf, shape=(self.p - 2, self.lookback_window_size), dtype=np.float32),
                "account": spaces.Box(low=-np.inf, high=np.inf, shape=(7, ), dtype=np.float32)
            })
        else:
            # HARDCODED P CHANGE THIS
            self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = ( self.p * self.lookback_window_size + 12 * self.lookback_window_size + 7, ), dtype=np.float32)

        # set randomization seed
        self.seed = seed
        random.seed(self.seed)

        # initialize state with the first candle
        self.state = self._next_observation()

    def _next_observation(self) -> dict:
        """
        :return: return a dictionary of the observed space {"price": pd.DataFrame, "account" pd.DataFrame}
        """
        # add previous position
        self.account_info = np.array([self.__dict__[col] for col in self.account_info_cols], dtype=np.float32)

        price_info = np.array([
            self.df.loc[self.current_step - self.lookback_window_size: (self.current_step - 1), col].values
            for col in self.price_info_cols
        ], dtype=np.float32)

        self.technical = np.array([
            self.df.loc[self.current_step - self.lookback_window_size: (self.current_step - 1), col].values
            for col in self.technical_cols
        ], dtype=np.float32)


        # add randomness to the data 0.01 std by default
        self.technical += self.noise * np.random.normal(0, 1, self.technical.shape)
        self.state = {
            "account": self.account_info,
            "price": price_info,
            "technical": self.technical,
        }

        # vectorize the dictionary
        if self.vectorize: self.state = self._vectorize_obs(self.state)

        return self.state

    @property
    def feature_names(self):
        return (self.account_info_cols +
                [
                    f"{col}__lag_{i}"
                    for col in self.price_info_cols + self.technical_cols
                    for i in range(self.lookback_window_size - 1, -1, -1)
                ]
                )

    def _vectorize_obs(self, d):
        # used to turn the dictionary observation space into a numpy vector
        x = []

        for i in d.values():
            if type(i) == np.float32:
                x.append(i)
            else:
                for j in i:
                    if type(j) == np.float32:
                        x.append(j)
                    else:
                        for k in j:
                            x.append(k)

        return np.asarray([x])


    def _take_action(self, decision):
        """
            take an action of 0, 1, 2 where 2 means maximum long position, 1 means neutral position and 0 maximum short
        """
        # assign categorical variable to position
        if decision == 0:
            action = -1
        elif decision == 1:
            action = 0
        elif decision == 2:
            action = 1

        # sample the current price between low and high to introduce randomization in the simulator
        # in order to avoid RL agent remember the training set
        current_price = random.uniform(
            self.df.loc[self.current_step, 'Low'], self.df.loc[self.current_step, 'High']
        )

        # get current_date "Date" as one column in the pandas df
        print(self.df.loc[self.current_step, 'Date'])
        print(type(self.df.loc[self.current_step, 'Date']))
        
        try:
            self.current_date = self.df.loc[self.current_step, 'Date'].strftime("%Y-%m-%d")
        except:
            self.current_date = self.df.loc[self.current_step, 'Date']
        # book-keeping previous volume held
        self.prev_volume_held = self.volume_held
        # current volume held after the agent takes an action, constraint [-max_num_volume, max_num_volume]
        self.volume_held = int(action * self.max_num_volume)

        # compute daily_position because of the changes in the volume_held
        self.daily_position = self.volume_held - self.prev_volume_held

        # hold current position (no position changes)
        if np.abs(self.daily_position) == 0:
            pass   # continue to calculate the unrealized PnL

        # when daily_position is the opposite side of volume_held
        elif np.sign(self.daily_position) != np.sign(self.prev_volume_held):
            # the action direction is the opposite of our current position
            # volume that can be closed out or close_volume
            close_volume = np.sign(self.daily_position) * np.minimum(abs(self.daily_position), abs(self.prev_volume_held))

            # If a part of previous position closed, then we calculate the realized PnL
            # Calculated Realized PnL: For the purpose of calculating Realized PnL, the cost basis does not change
            self.realized_pnl += (current_price - self.cost_basis + np.sign(self.daily_position) * self.transaction_cost) * (-close_volume)

            # if volume_held is 0, close our previous long or short position, then clear the cost_basis
            if int(self.volume_held) == 0:
                self.cost_basis = 0.

            if DEBUG:
                # debug statement and print out number of contracts closed before going the opposite side of the direction
                if close_volume > 0:
                    print("%10s, Action %6.3f, %6s %8d contracts at $%10.3f (Vol %5d contracts at $%10.3f)" %
                          (self.current_date, action, "Close", close_volume, current_price, self.prev_volume_held + close_volume, self.cost_basis))
                elif close_volume < 0:
                    print("%10s, Action %6.3f, %6s %8d contracts at $%10.3f (Vol %5d contracts at $%10.3f)" %
                          (self.current_date, action, "Close", close_volume, current_price, self.prev_volume_held + close_volume, self.cost_basis))

            if abs(close_volume) < abs(self.prev_volume_held):
                # if position closed is smaller than existing position then cost_basis stays the same
                pass
            else:
                # establish a new positive position at volume_held
                self.cost_basis = current_price + self.transaction_cost

                if DEBUG:
                    if self.volume_held > 0:
                        print("%10s, Action %6.3f, %6s %8d contracts at $%10.3f (Vol %5d contracts at $%10.3f)" %
                              (self.current_date, action, "Buy", self.volume_held, current_price, self.volume_held, self.cost_basis))
                    elif self.volume_held < 0:
                        print("%10s, Action %6.3f, %6s %8d contracts at $%10.3f (Vol %5d contracts at $%10.3f)" %
                              (self.current_date, action, "Sell", self.volume_held, current_price, self.volume_held, self.cost_basis))

        # accumulation of current position
        else:
            # the action direction is the same as our current position
            prev_cost = self.cost_basis * self.prev_volume_held

            # calculate additional cost and update new cost_basis
            additional_cost = self.daily_position * (current_price + np.sign(self.daily_position) * self.transaction_cost)

            # 1e-8 avoids 0 / 0 when starting the simulator
            self.cost_basis = (prev_cost + additional_cost) / (self.volume_held + 1e-8)

            if DEBUG:
                if self.daily_position > 0:
                    print("%10s, Action %6.3f, %6s %8d contracts at $%10.3f (Vol %5d contracts at $%10.3f)" %
                          (self.current_date, action, "Buy", self.daily_position, current_price, self.volume_held, self.cost_basis))
                elif self.daily_position < 0:
                    print("%10s, Action %6.3f, %6s %8d contracts at $%10.3f (Vol %5d contracts at $%10.3f)" %
                          (self.current_date, action, "Sell", self.daily_position, current_price, self.volume_held, self.cost_basis))

        # update the total of volume transacted
        self.total_volume_transacted += abs(self.daily_position)

        # book keeping the daily_value associated with self.daily_position
        self.daily_value = self.daily_position * current_price

        # calculate unrealized pnl or the accumulate returns from the start
        self.unrealized_pnl = self.volume_held * (current_price - self.cost_basis)
        self.yesterday_m2m = self.m2m
        self.m2m = self.realized_pnl + self.unrealized_pnl
        self.daily_pnl = self.m2m - self.yesterday_m2m

        if self.m2m > self.max_net_worth:
            self.max_net_worth = self.m2m

        # append today's pnl
        self.historical_daily_pnl.append(self.daily_pnl)

        # compute Sharpe ratio after observing 10 bars
        if len(self.historical_daily_pnl) > 10:
            self.sharpe_ratio = (np.mean(self.historical_daily_pnl) / (1e-8 + np.std(self.historical_daily_pnl))) * np.sqrt(252)

    def step(self, action) -> tuple:
        # Execute one time step within environment
        self._take_action(action)
        # reward = self.daily_pnl

        # Moody's reward function MU * (A_{t-1} * r_t - self.transaction_cost * abs(A_t - A_{t-1}))
        # where r_t = p_t - p_{t-1}

        # difference between the current close price and previous close price
        r_t = self.df.loc[self.current_step, "Close"] - self.df.loc[self.current_step - 1, "Close"]
        self.reward = MU * self.prev_action * r_t - self.transaction_cost * abs(action - self.prev_action)

        # keep track of the previous position to calculate the rewards (to penalize large turnover)
        self.prev_action = action

        self.current_step += 1
        self.done = self.current_step >= self.nT - 1


        # # save results when done
        # if done and self.file_name is not None:
        #     self.output.to_csv("%s/%s.csv" % (self.prefix, self.file_name), index=False, float_format='%.3f')

        next_obs = self._next_observation()
        return next_obs, self.reward, self.done, {}

    def reset(self):
        # reset the state of the environment to an initial state
        # self.balance = self.initial_account_balance  # balance of account or available cash
        self.volume_held = 0.        # number of contracts an agent held in the current step
        self.prev_volume_held = 0.   # number of contracts an agent held in the previous step
        self.prev_action = 0.        # action [-1, 1] an agent takes in the previous step

        self.cost_basis = 0.         # averaged cost for a contract
        self.total_volume_transacted = 0.  # total number of contracts traded
        self.max_net_worth = 0             # reset maximum net worth
        self.current_step = self.lookback_window_size  # reset current step size

        self.realized_pnl = 0.
        self.unrealized_pnl = 0.
        self.m2m = 0.  # market to market https://www.investopedia.com/terms/m/marktomarket.asp
        self.yesterday_m2m = 0.
        self.daily_pnl = 0.
        self.sharpe_ratio = 0.
        self.daily_position = 0.
        self.daily_value = 0.
        self.account_info = None  # clear the dictionary to book keeping current account
        self.historical_daily_pnl = []

        self.reward = 0  # immediate reward
        self.done = False  # check if simulation terminates or not
        # create result file
        if self.file_name is not None:
            with open("%s/%s.csv" % (self.prefix, self.file_name), 'w', newline='') as file:
                writer = csv.writer(file, delimiter=",")
                writer.writerow(['Date', 'Open', 'Close', 'High', 'Low', 'Volume',
                                    'HmL', 'LmO', 'HmO', 'CmCtm1', 'adj_close', 'adj_1mr', 'adj_2mr', 'adj_3mr', 'adj_12mr', 'macd', 'prsi', 'vrsi',
                                    'Daily action', 'Volume held', 'Cost basis for volume held', 'Total transacted value',
                                  'Unrealized value', 'Realized value', 'Mark to Market', 'Daily Profit', 'Shape Ratio'])
        return self._next_observation()

    def _render_to_file(self):
        # write the simulation result to a file if file_name is not None
        if self.file_name is not None:
            results_info = self.df.loc[self.current_step].values.tolist() + \
                          [self.daily_position, self.volume_held, self.cost_basis,
                           self.daily_value, self.unrealized_pnl, self.realized_pnl, self.m2m, self.daily_pnl, self.sharpe_ratio]

            # round timestamp to date if daily data was feed
            results_info[0] = results_info[0].strftime("%Y-%m-%d %H:%M:%S")

            # round all digits to 3
            for i in range(1, len(results_info)):
                results_info[i] = '{0:10.3f}'.format(results_info[i])

            with open('%s/%s.csv' % (self.prefix, self.file_name), 'a+', newline='') as file:
                writer = csv.writer(file, delimiter=",")
                writer.writerow(results_info)

    def render(self, mode='live', **kwargs):
        # Render the environment to the screen
        if self.file_name is not None:
            self._render_to_file()

        if mode == "debug":
            self.render_text()

        elif mode == 'live':
            if self.visualization is None:
                self.visualization = MarketSimulatorGraphics(
                    self.df, kwargs.get('title', None))

            if self.current_step > self.lookback_window_size:
                self.visualization.render(
                    self.current_step, self.m2m, self.trades, window_size=self.lookback_window_size)

    def render_text(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: %d' % self.current_step)
        print(f'Realized Profit: %.3f' % self.realized_pnl)
        print(f'Unrealized Profit: %.3f' % self.unrealized_pnl)
        print(f'Net worth: %.3f (Max net worth: %.3f)' % (self.m2m, self.max_net_worth))
        print(f'Sharpe Ratio: %.3f' % self.sharpe_ratio)
        print(f'Volume held: %d (Total transacted: %d)' % (self.volume_held, self.total_volume_transacted))
        print(f'Avg cost for volume held: %.3f (Total sales value: %.3f)' % (self.cost_basis, self.daily_value))

    def get_account(self):
        # return account information as a dictionary
        return {
            "date": self.df.loc[self.current_step, "Date"].strftime("%Y-%m-%d %H:%M:%S"),
            "realized_pnl": round(self.realized_pnl, 3),
            "unrealized_pnl": round(self.unrealized_pnl, 3),
            "max_net_worth": round(self.max_net_worth, 3),
            "volume_held": round(self.volume_held, 3),
            "total_volume_transacted": round(self.total_volume_transacted, 3),
            "cost_basis": round(self.cost_basis, 3),
            "sharpe_ratio": round(self.sharpe_ratio, 3)
        }

    def close(self):
        if self.visualization is not None:
            self.visualization.close()
            self.visualization = None


    
    