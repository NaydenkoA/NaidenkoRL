from torch_model import reccurentActor
from torch_learning_utils import train_model
import pandas as pd

def lr_schedual(frac):
        if frac < 0.5:
            return 2.5e-4*frac/0.5 + 1e-4
        else:
            return 3.5e-4
        
def epsilon_schedual(frac):
    if frac < 0.25:
        return 0.1
    else:
        if frac < 0.6:
            return (0.01 - 0.1)/(0.6 - 0.25)*(frac - 0.25) + 0.1 
        else: 
            return 0.01

def eff_entropy_schedual(frac):
    if frac < 0.5:
        return (0.5 - frac)/0.5*0.005 + 0.001
    else:
        return 0.001

def create_config(
        top_laverage, 
        bottom_laverage,
        nect_price,
        nav,
        volatile_part,
        vol_token_price
    ):
    return {
        'nect_price': nect_price,
        'nav': nav,
        'vol_part': volatile_part,
        'vol_token_honey_price': vol_token_price,
        'is_reversed': True,
        'top_leverage': max(top_laverage, bottom_laverage),
        'bottom_leverage': min(top_laverage, bottom_laverage)
    }

model = reccurentActor()
model.set_initial_state('actor_weights.pth')

df = pd.read_csv('wbtc_train_data_expanded.csv')
nav = df['nav'].values
nect_price = df['nect_price'].values
vol_token_price = df['wbtc_price'].values
volatile_part = df['wbtc_frac'].values

nav = nav[::4]
nect_price = nect_price[::4]
vol_token_price = vol_token_price[::4]
volatile_part = volatile_part[::4]

configs = [
    create_config(1.8, 2.2, nect_price=nect_price, nav=nav, vol_token_price=volatile_part, volatile_part=volatile_part),
    create_config(2.3, 2.7, nect_price=nect_price, nav=nav, vol_token_price=volatile_part, volatile_part=volatile_part),
    create_config(2.8, 3.2, nect_price=nect_price, nav=nav, vol_token_price=volatile_part, volatile_part=volatile_part),
    create_config(1.8, 2.2, nect_price=nect_price, nav=nav, vol_token_price=volatile_part, volatile_part=volatile_part),
    create_config(2.3, 2.7, nect_price=nect_price, nav=nav, vol_token_price=volatile_part, volatile_part=volatile_part),
    create_config(2.8, 3.2, nect_price=nect_price, nav=nav, vol_token_price=volatile_part, volatile_part=volatile_part)
]

if __name__ == "__main__":

    logs = train_model(
        model=model,
        configs=configs,
        n_episodes = 200,
        lr = lr_schedual,
        ent_coef_alt = eff_entropy_schedual,
        epsilon = epsilon_schedual
    )

    df = pd.DataFrame(logs)
    df.to_csv('torch_training_logs.csv', index=False)