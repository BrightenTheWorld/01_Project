import pandas as pd
import Graph_response_curves as Rc

df = Rc.df
phi = Rc.phi

df.to_csv(r"C:\Users\User\OneDrive - 공주대학교\지반연구실 Geotech\GeoTech\df.csv", mode='w')
phi.to_csv(r"C:\Users\User\OneDrive - 공주대학교\지반연구실 Geotech\GeoTech\phi.csv", mode='w')
