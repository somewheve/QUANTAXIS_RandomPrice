#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import click
import random
import copy
from QUANTAXIS.QAUtil.QADate_trade import QA_util_get_last_day
from QUANTAXIS.QAUtil.QADate import QA_util_date_int2str, QA_util_date_str2int


class OrnsteinUhlenbeckActionNoise:

    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)

        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


time_index = pd.timedelta_range('21:00:00.500000', '23:00:00', freq='500ms').tolist() + \
             pd.timedelta_range('09:00:00.500000', '10:15:00', freq='500ms').tolist() + \
             pd.timedelta_range('10:30:00.500000', '11:30:00', freq='500ms').tolist() + \
             pd.timedelta_range('13:30:00.500000', '15:00:00', freq='500ms').tolist()


def get_random_price(price, code='rb1905', tradingDay='20181119', mu=0, sigma=0.2, theta=0.15, dt=1e-2, ifprint=False,
                     weight=0.1, callback=None):
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.array(mu))

    data = []
    tick_pickle = {
        'InstrumentID': str(code),
        'UpdateTime': '',
        'UpdateMillisec': '',
        'LastPrice': '',
        'Volume': 0,
        'TradingDay': tradingDay,
        'ActionDay': QA_util_date_str2int(QA_util_get_last_day(QA_util_date_int2str(tradingDay)))
    }

    for item in time_index:
        x = str(item).split()[2]
        if '.' in x:
            tick_pickle['UpdateMillisec'] = 500
        else:
            tick_pickle['UpdateMillisec'] = 0

        tick_pickle['UpdateTime'] = x.split('.')[0]
        if item.seconds >= 75600:
            tick_pickle['ActionDay'] = QA_util_date_str2int(
                QA_util_get_last_day(QA_util_date_int2str(tick_pickle['TradingDay'])))
        else:
            tick_pickle['ActionDay'] = tick_pickle['TradingDay']
        tick_pickle['Volume'] += random.randint(50, 5000)
        tick_pickle['LastPrice'] = (ou_noise() + 1) * \
                                   weight * price + (1 - weight) * price
        data.append(copy.deepcopy(tick_pickle))
        if ifprint:
            print(tick_pickle)
        if callback is not None:
            callback(tick_pickle)
    return pd.DataFrame(data)


@click.command()
@click.option('--price', default=3600)
@click.option('--code', default='rb1905')
@click.option('--tradingday', default='20181119')
@click.option('--mu', default=0)
@click.option('--sigma', default=0.2)
@click.option('--theta', default=0.15)
@click.option('--dt', default=1e-2)
@click.option('--ifprint', default=True)
def generate(price, code, tradingday, mu, sigma, theta, dt, ifprint):
    data = get_random_price(price, code, tradingday,
                            mu, sigma, theta, dt, ifprint)
    print(data)
    data.LastPrice.plot()
    plt.show()


def get_future_tick_format(tick) -> dict:
    """ 生成期货 tick"""
    return dict(
        symbol=tick["InstrumentID"],
        exchange="QASIM",
        local_symbol=f"{tick['InstrumentID']}.QASIM",
        last_price=tick['LastPrice'],
        volume=tick["Volume"],
        datetime=f'{QA_util_date_int2str(tick["TradingDay"])} {tick["UpdateTime"]}.{tick["UpdateMillisec"]}',
        name=tick["InstrumentID"],
        low_price=0.0,
        gateway_name="ctp",
        open_interest=0.0,
        open_price=0.0,
        limit_up=0.0,
        limit_down=0.0,
        last_volume=0.0,
        high_price=0.0,
        ask_price_1=tick["LastPrice"] + random.randint(1, 3),
        ask_price_2=0,
        ask_price_3=0,
        ask_price_4=0,
        ask_price_5=0,
        ask_volume_1=tick["Volume"] + random.randint(-3, 5),
        ask_volume_2=0,
        ask_volume_3=0,
        ask_volume_4=0,
        ask_volume_5=0,
        average_price=0.0,
        bid_price_1=tick["LastPrice"] - random.randint(1, 3),
        bid_price_2=0,
        bid_price_3=0,
        bid_price_4=0,
        bid_price_5=0,
        bid_volume_1=tick["Volume"] + random.randint(-3, 5),
        bid_volume_2=0,
        bid_volume_3=0,
        bid_volume_4=0,
        bid_volume_5=0,
        turnover=0.0,
        settlement_price=0.0,
        pre_settlement_price=0.0,
        pre_open_interest=0.0,
        pre_close=0.0,
    )


if __name__ == '__main__':
    print(get_random_price(3600))
