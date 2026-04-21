import random
import string
import datetime
import math
from pathlib import Path

CURRENT_PRICE = 100
CURRENT_VOL = 0.05
CURRENT_VAR = CURRENT_VOL**2

EXPECTED_VOL = 0.04
STEPS_PER_DAY = 100
dt = 1/STEPS_PER_DAY

def increment_price(dt = dt, drift = 0.05, z = None):
    global CURRENT_PRICE, CURRENT_VOL
    if z is None:
        z = random.gauss(0, 1)
    exponent = (drift - 0.5*(CURRENT_VAR))*dt + math.sqrt(CURRENT_VAR*dt)*z
    CURRENT_PRICE *= math.exp(exponent)
    return None

def increment_variance(
        dt = dt,
        kappa = 3.0,    # the speed at which v reverts to theta
        theta = 0.1,   # long term variance
        xi = 0.05,       # vol of vol
        z = None
    ):
    global CURRENT_VOL, CURRENT_VAR
    if z is None:
        z = random.gauss(0, 1)
    time_increment = kappa*(theta - CURRENT_VAR)*dt
    random_increment = xi*math.sqrt(max(0, CURRENT_VAR))*math.sqrt(dt)*z

    dv = time_increment + random_increment
    CURRENT_VAR = max(CURRENT_VAR + dv, 0)
    CURRENT_VOL = math.sqrt(CURRENT_VAR)

    return None

def heston_step(
        dt = dt,
        kappa = 3.0,
        theta = 0.04,
        xi = 0.05,
        mu = 0.001, # drift term
        rho = -0.7 # correlation of variance and price
    ):
    z_price = random.gauss(0, 1)
    z_perp = random.gauss(0, 1)
    z_vol = rho * z_price + math.sqrt(1 - rho**2)*z_perp

    increment_price(dt, drift = mu, z=z_price)
    increment_variance(dt, kappa, theta, xi, z = z_vol)

    return None

def main():
    global CURRENT_PRICE, CURRENT_VOL, CURRENT_VAR

    steps = int(1e4)
    bar_size = 10  # steps per OHLC candle

    current_time = 0.0
    script_dir = Path(__file__).parent

    events_path = script_dir / "events.csv"
    ohlcv_path = script_dir / "ohlcv.csv"

    with open(events_path, 'w') as events, open(ohlcv_path, 'w') as ohlcv:

        events.write("time,price,vol,variance\n")
        ohlcv.write("time,open,high,low,close,volume\n")

        # OHLC state
        open_price = CURRENT_PRICE
        high_price = CURRENT_PRICE
        low_price = CURRENT_PRICE
        volume = 0

        for i in range(steps):

            heston_step()
            current_time += dt

            events.write(f"{current_time},{CURRENT_PRICE},{CURRENT_VOL},{CURRENT_VAR}\n")

            high_price = max(high_price, CURRENT_PRICE)
            low_price = min(low_price, CURRENT_PRICE)
            close_price = CURRENT_PRICE
            volume += abs(random.gauss(0, 1))

            if (i + 1) % bar_size == 0:
                ohlcv.write(
                    f"{current_time},{open_price},{high_price},{low_price},{close_price},{volume}\n"
                )
                open_price = CURRENT_PRICE
                high_price = CURRENT_PRICE
                low_price = CURRENT_PRICE
                volume = 0

if __name__ == "__main__":
    main()