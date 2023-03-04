{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c7e0f11d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn import metrics\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "eb1408dd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>stn_code</th>\n",
       "      <th>sampling_date</th>\n",
       "      <th>state</th>\n",
       "      <th>location</th>\n",
       "      <th>agency</th>\n",
       "      <th>type</th>\n",
       "      <th>so2</th>\n",
       "      <th>no2</th>\n",
       "      <th>pm10</th>\n",
       "      <th>pm2_5</th>\n",
       "      <th>location_monitoring_station</th>\n",
       "      <th>date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>150.0</td>\n",
       "      <td>February - M021990</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>4.8</td>\n",
       "      <td>17.4</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2/1/1990</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>151.0</td>\n",
       "      <td>February - M021990</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Industrial Area</td>\n",
       "      <td>3.1</td>\n",
       "      <td>7.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2/1/1990</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>152.0</td>\n",
       "      <td>February - M021990</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>6.2</td>\n",
       "      <td>28.5</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2/1/1990</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>150.0</td>\n",
       "      <td>March - M031990</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>6.3</td>\n",
       "      <td>14.7</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3/1/1990</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>151.0</td>\n",
       "      <td>March - M031990</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Industrial Area</td>\n",
       "      <td>4.7</td>\n",
       "      <td>7.5</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3/1/1990</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  stn_code       sampling_date           state   location agency  \\\n",
       "0    150.0  February - M021990  Andhra Pradesh  Hyderabad    NaN   \n",
       "1    151.0  February - M021990  Andhra Pradesh  Hyderabad    NaN   \n",
       "2    152.0  February - M021990  Andhra Pradesh  Hyderabad    NaN   \n",
       "3    150.0     March - M031990  Andhra Pradesh  Hyderabad    NaN   \n",
       "4    151.0     March - M031990  Andhra Pradesh  Hyderabad    NaN   \n",
       "\n",
       "                                 type  so2   no2  pm10  pm2_5  \\\n",
       "0  Residential, Rural and other Areas  4.8  17.4   NaN    NaN   \n",
       "1                     Industrial Area  3.1   7.0   NaN    NaN   \n",
       "2  Residential, Rural and other Areas  6.2  28.5   NaN    NaN   \n",
       "3  Residential, Rural and other Areas  6.3  14.7   NaN    NaN   \n",
       "4                     Industrial Area  4.7   7.5   NaN    NaN   \n",
       "\n",
       "  location_monitoring_station      date  \n",
       "0                         NaN  2/1/1990  \n",
       "1                         NaN  2/1/1990  \n",
       "2                         NaN  2/1/1990  \n",
       "3                         NaN  3/1/1990  \n",
       "4                         NaN  3/1/1990  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.read_csv('data.csv', encoding='unicode_escape')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "636bd7b4",
   "metadata": {},
   "source": [
    "stn_code: Station Code\n",
    "sampling_date: Date of sampling (note how this is formatted)\n",
    "state: State\n",
    "location: Location of recording\n",
    "agency: Agency\n",
    "type: Type of area\n",
    "so2: Sulphur dioxide (µg/m3)\n",
    "no2: Nitrogen dioxide (µg/m3)\n",
    "pm10: Respirable Suspended Particulate Matter (µg/m3)\n",
    "pm2_5: Suspended Particulate Matter (µg/m3)\n",
    "location_monitoring_station: Location of data collection\n",
    "date: Date of sampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9860a0a8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(435742, 12)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "9c9bc460",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 435742 entries, 0 to 435741\n",
      "Data columns (total 12 columns):\n",
      " #   Column                       Non-Null Count   Dtype  \n",
      "---  ------                       --------------   -----  \n",
      " 0   stn_code                     291665 non-null  object \n",
      " 1   sampling_date                435739 non-null  object \n",
      " 2   state                        435742 non-null  object \n",
      " 3   location                     435739 non-null  object \n",
      " 4   agency                       286261 non-null  object \n",
      " 5   type                         430349 non-null  object \n",
      " 6   so2                          401096 non-null  float64\n",
      " 7   no2                          419509 non-null  float64\n",
      " 8   pm10                         395520 non-null  float64\n",
      " 9   pm2_5                        198355 non-null  float64\n",
      " 10  location_monitoring_station  408251 non-null  object \n",
      " 11  date                         435735 non-null  object \n",
      "dtypes: float64(4), object(8)\n",
      "memory usage: 39.9+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "bd8c2a06",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "stn_code                       144077\n",
       "sampling_date                       3\n",
       "state                               0\n",
       "location                            3\n",
       "agency                         149481\n",
       "type                             5393\n",
       "so2                             34646\n",
       "no2                             16233\n",
       "pm10                            40222\n",
       "pm2_5                          237387\n",
       "location_monitoring_station     27491\n",
       "date                                7\n",
       "dtype: int64"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "84c9172f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>so2</th>\n",
       "      <th>no2</th>\n",
       "      <th>pm10</th>\n",
       "      <th>pm2_5</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>401096.000000</td>\n",
       "      <td>419509.000000</td>\n",
       "      <td>395520.000000</td>\n",
       "      <td>198355.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>10.829414</td>\n",
       "      <td>25.809623</td>\n",
       "      <td>108.832784</td>\n",
       "      <td>220.783480</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>11.177187</td>\n",
       "      <td>18.503086</td>\n",
       "      <td>74.872430</td>\n",
       "      <td>151.395457</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>5.000000</td>\n",
       "      <td>14.000000</td>\n",
       "      <td>56.000000</td>\n",
       "      <td>111.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>8.000000</td>\n",
       "      <td>22.000000</td>\n",
       "      <td>90.000000</td>\n",
       "      <td>187.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>13.700000</td>\n",
       "      <td>32.200000</td>\n",
       "      <td>142.000000</td>\n",
       "      <td>296.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>909.000000</td>\n",
       "      <td>876.000000</td>\n",
       "      <td>6307.033333</td>\n",
       "      <td>3380.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 so2            no2           pm10          pm2_5\n",
       "count  401096.000000  419509.000000  395520.000000  198355.000000\n",
       "mean       10.829414      25.809623     108.832784     220.783480\n",
       "std        11.177187      18.503086      74.872430     151.395457\n",
       "min         0.000000       0.000000       0.000000       0.000000\n",
       "25%         5.000000      14.000000      56.000000     111.000000\n",
       "50%         8.000000      22.000000      90.000000     187.000000\n",
       "75%        13.700000      32.200000     142.000000     296.000000\n",
       "max       909.000000     876.000000    6307.033333    3380.000000"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "6d1f4973",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "stn_code                        803\n",
       "sampling_date                  5485\n",
       "state                            37\n",
       "location                        304\n",
       "agency                           64\n",
       "type                             10\n",
       "so2                            4197\n",
       "no2                            6864\n",
       "pm10                           6065\n",
       "pm2_5                          6668\n",
       "location_monitoring_station     991\n",
       "date                           5067\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.nunique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "70d26ddf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['stn_code', 'sampling_date', 'state', 'location', 'agency', 'type',\n",
       "       'so2', 'no2', 'pm10', 'pm2_5', 'location_monitoring_station', 'date'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "23d5c3d1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.PairGrid at 0x2c5577adf70>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x720 with 20 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.pairplot(data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9a1f0db8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Maharashtra                    60384\n",
       "Uttar Pradesh                  42816\n",
       "Andhra Pradesh                 26368\n",
       "Punjab                         25634\n",
       "Rajasthan                      25589\n",
       "Kerala                         24728\n",
       "Himachal Pradesh               22896\n",
       "West Bengal                    22463\n",
       "Gujarat                        21279\n",
       "Tamil Nadu                     20597\n",
       "Madhya Pradesh                 19920\n",
       "Assam                          19361\n",
       "Odisha                         19279\n",
       "Karnataka                      17119\n",
       "Delhi                           8551\n",
       "Chandigarh                      8520\n",
       "Chhattisgarh                    7831\n",
       "Goa                             6206\n",
       "Jharkhand                       5968\n",
       "Mizoram                         5338\n",
       "Telangana                       3978\n",
       "Meghalaya                       3853\n",
       "Puducherry                      3785\n",
       "Haryana                         3420\n",
       "Nagaland                        2463\n",
       "Bihar                           2275\n",
       "Uttarakhand                     1961\n",
       "Jammu & Kashmir                 1289\n",
       "Daman & Diu                      782\n",
       "Dadra & Nagar Haveli             634\n",
       "Uttaranchal                      285\n",
       "Arunachal Pradesh                 90\n",
       "Manipur                           76\n",
       "Sikkim                             1\n",
       "andaman-and-nicobar-islands        1\n",
       "Lakshadweep                        1\n",
       "Tripura                            1\n",
       "Name: state, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['state'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "af4e66db",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA4gAAAH+CAYAAADeVCA7AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAB9qklEQVR4nO3debzt5dz/8de70qySkpScEO6UUqdkLqEMyRQZk8icyE2mn+lOhhs3oVuEFJIhCqWkZGhQKg3qLgknKVIpKU7evz+ua3XW2Wfvc87+Dnvt4f18PPZjr/Vde33Wtfdew/dzDZ9LtomIiIiIiIhYYdQNiIiIiIiIiOkhCWJEREREREQASRAjIiIiIiKiSoIYERERERERQBLEiIiIiIiIqJIgRkREREREBAArjboBU2299dbzvHnzRt2MJfz9739njTXWmJMx+4qbmIk53eMm5tyM2VfcxJybMfuKm5iJOd3jzuWYXTj33HP/Ynv9cW+0Pae+tt12W09Hp5566pyN2VfcxEzM6R43MedmzL7iJubcjNlX3MRMzOkedy7H7AJwjifIlzLFNCIiIiIiIoCsQYyIiIiIiIgqCWJEREREREQASRAjIiIiIiKiSoIYERERERERQBLEiIiIiIiIqJIgRkREREREBJAEMSIiIiIiIqreEkRJD5R0/tDX3yTtL2ldSSdLurx+v9vQfd4q6QpJl0naZej4tpIurLd9QpLq8VUkfa0eP0vSvL5+n4iIiIiIiNmutwTR9mW2t7a9NbAtcCtwLHAgcIrtzYBT6nUkbQ7sCTwY2BX4tKQVa7hDgX2BzerXrvX4PsANtu8PfAz4YF+/T0RERERExGw3VVNMdwZ+Y/t3wO7AEfX4EcDT6+XdgaNt3277t8AVwPaSNgTWsn2GbQNfGnOfQaxvADsPRhcjIiIiIiJicqYqQdwT+Gq9vIHtawDq93vU4xsBfxi6z4J6bKN6eezxxe5jeyFwE3D3HtofEREREREx66kMyvX4ANLKwB+BB9u+VtKNttcZuv0G23eT9CngDNtH1eOHA98Hfg8cbPvx9fijgTfb3k3SxcAuthfU234DbG/7+jFt2JcyRZUNNthg26OPPrrX37mJW265hTXXXHNOxuwrbmIm5nSPm5hzM2ZfcRNzbsbsK25iJuZ0jzuXY3Zhp512Otf2/HFvtN3rF2Ua6ElD1y8DNqyXNwQuq5ffCrx16Od+ADy8/sylQ8efB3xm+Gfq5ZWAv1CT3om+tt12W09Hp5566pyN2VfcxEzM6R43MedmzL7iJubcjNlX3MRMzOkedy7H7AJwjifIl1bqISEd63ksml4KcBywF/CB+v07Q8e/IumjwL0oxWjOtn2HpJsl7QCcBbwYOGRMrDOAZwM/qr9wRETEtHXh1TfxkgO/12nMA7ZcOGtjXvWBp3TahoiImFivCaKk1YEnAK8YOvwB4BhJ+1Cmj+4BYPtiSccAlwALgdfYvqPe51XAF4HVgBPqF8DhwJGSrgD+SlnrGBEREREREQ30miDavpUxRWNc1gfuPMHPHwQcNM7xc4Atxjl+GzXBjIiIiIiIiHamqoppRERERERETHNJECMiIiIiIgJIghgRERERERFVEsSIiIiIiIgAkiBGRERERERElQQxIiIiIiIigCSIERERERERUSVBjIiIiIiICCAJYkRERERERFRJECMiIiIiIgJIghgRERERERFVEsSIiIiIiIgAkiBGRERERERElQQxIiIiIiIigCSIERERERERUSVBjIiIiIiICCAJYkRERERERFRJECMiIiIiIgJIghgRERERERFVEsSIiIiIiIgAkiBGRERERERElQQxIiIiIiIigCSIERERERERUSVBjIiIiIiICCAJYkRERERERFRJECMiIiIiIgJIghgRERERERFVEsSIiIiIiIgAkiBGRERERERElQQxIiIiIiIigCSIERERERERUSVBjIiIiIiICABWGnUDIiJiZpt34PcmfZ8DtlzISxrcbzbELHE7DxkREdGJjCBGREREREQEkAQxIiIiIiIiqiSIERERERERAfScIEpaR9I3JF0q6deSHi5pXUknS7q8fr/b0M+/VdIVki6TtMvQ8W0lXVhv+4Qk1eOrSPpaPX6WpHl9/j4RERERERGzWd8jiB8HTrT9IGAr4NfAgcAptjcDTqnXkbQ5sCfwYGBX4NOSVqxxDgX2BTarX7vW4/sAN9i+P/Ax4IM9/z4RERERERGzVm8JoqS1gMcAhwPY/qftG4HdgSPqjx0BPL1e3h042vbttn8LXAFsL2lDYC3bZ9g28KUx9xnE+gaw82B0MSIiIiIiIianzxHE+wJ/Br4g6TxJn5O0BrCB7WsA6vd71J/fCPjD0P0X1GMb1ctjjy92H9sLgZuAu/fz60RERERERMxuKoNyPQSW5gNnAo+0fZakjwN/A15ne52hn7vB9t0kfQo4w/ZR9fjhwPeB3wMH2358Pf5o4M22d5N0MbCL7QX1tt8A29u+fkxb9qVMUWWDDTbY9uijj+7ld27jlltuYc0115yTMfuKm5iJOd3jzpaYF15906RjbrAaXPuPNq2auTH7ijubY2650dpLvX22vJamU9zETMzpHncux+zCTjvtdK7t+ePdtlKPj7sAWGD7rHr9G5T1htdK2tD2NXX66HVDP3/voftvDPyxHt94nOPD91kgaSVgbeCvYxti+zDgMID58+d7xx13bP/bdey0006j63bNlJh9xU3MxJzucWdLzCYbyR+w5UI+cmG3H0EzJWZfcWdzzKtesONSb58tr6XpFDcxE3O6x53LMfvW2xRT238C/iDpgfXQzsAlwHHAXvXYXsB36uXjgD1rZdJNKcVozq7TUG+WtENdX/jiMfcZxHo28CP3NSQaERERERExy/U5ggjwOuDLklYGrgT2piSlx0jahzJ9dA8A2xdLOoaSRC4EXmP7jhrnVcAXgdWAE+oXlAI4R0q6gjJyuGfPv09ERERERMSs1WuCaPt8YLy5rTtP8PMHAQeNc/wcYItxjt9GTTAjIiIiIiKinb73QYyIiIiIiIgZIgliREREREREAEkQIyIiIiIiokqCGBEREREREUASxIiIiIiIiKiSIEZERERERASQBDEiIiIiIiKqJIgREREREREBJEGMiIiIiIiIKgliREREREREAEkQIyIiIiIiokqCGBEREREREUASxIiIiIiIiKiSIEZERERERASQBDEiIiIiIiKqJIgREREREREBJEGMiIiIiIiIKgliREREREREAEkQIyIiIiIiokqCGBEREREREUASxIiIiIiIiKiSIEZERERERASQBDEiIiIiIiKqJIgREREREREBJEGMiIiIiIiIKgliREREREREAEkQIyIiIiIiokqCGBEREREREUASxIiIiIiIiKiSIEZERERERASQBDEiIiIiIiKqJIgREREREREBJEGMiIiIiIiIKgliREREREREALDSqBsQERERsTTzDvzeUm8/YMuFvGQZPzNZMyXmeHGv+sBTOn+MiJg7eh1BlHSVpAslnS/pnHpsXUknS7q8fr/b0M+/VdIVki6TtMvQ8W1rnCskfUKS6vFVJH2t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joiIiIiImGuWdx/ESZO0fh05RNJqwOOBS4HjgL3qj+0FfKdePg7Ys1Ym3ZRSjOZs29cAN0vaoa4vfPGY+wxiPRv4kZc2ZzYiIiIiIiIm1P08h0U2BI6o6xBXAI6x/V1JZwDHSNoH+D2wB4DtiyUdA1wCLAReU6eoArwK+CKwGmVd4gn1+OHAkZKuoIwc7tnj7xMRERERETGr9ZYg2v4V8NBxjl8P7DzBfQ4CDhrn+DnAEusXbd9GTTAjIiIiIiKind6mmEZERERERMTMkgQxIiIiIiIigCSIERERERERUSVBjIiIiIiICCAJYkRERERERFRJECMiIiIiIgJIghgRERERERFVEsSIiIiIiIgAkiBGRERERERElQQxIiIiIiIigCSIERERERERUSVBjIiIiIiICCAJYkRERERERFRJECMiIiIiIgJIghgRERERERFVEsSIiIiIiIgAkiBGRERERERElQQxIiIiIiIigCSIERERERERUSVBjIiIiIiICCAJYkRERERERFRJECMiIiIiIgJIghgRERERERFVEsSIiIiIiIgAkiBGRERERERElQQxIiIiIiIigCSIERERERERUSVBjIiIiIiICABWGnUDIiIiIiJiZph34PcmfZ8DtlzISxrcbzbE/OKua3QabypkBDEiIiIiIiKAJIgRERERERFRJUGMiIiIiIgIIAliREREREREVEkQIyIiIiIiAkiCGBEREREREVVvCaKke0s6VdKvJV0s6fX1+LqSTpZ0ef1+t6H7vFXSFZIuk7TL0PFtJV1Yb/uEJNXjq0j6Wj1+lqR5ff0+ERERERERs12fI4gLgQNs/wewA/AaSZsDBwKn2N4MOKVep962J/BgYFfg05JWrLEOBfYFNqtfu9bj+wA32L4/8DHggz3+PhEREREREbNabwmi7Wts/7Jevhn4NbARsDtwRP2xI4Cn18u7A0fbvt32b4ErgO0lbQisZfsM2wa+NOY+g1jfAHYejC5GRERERETE5EzJGsQ69fOhwFnABravgZJEAveoP7YR8Iehuy2oxzaql8ceX+w+thcCNwF37+WXiIiIiIiImOVUBuV6fABpTeDHwEG2vyXpRtvrDN1+g+27SfoUcIbto+rxw4HvA78HDrb9+Hr80cCbbe8m6WJgF9sL6m2/Aba3ff2YNuxLmaLKBhtssO3RRx/d6+/cxHV/vYlr/9FtzA1WY0bEbBJ3y43WXubP3HLLLay55potWjV7Yl549U2tY86U59N0eY4mZmJOddzEnJsx+4qbmIk53ePOlJibrr1i5+d5Xdhpp53OtT1/vNtW6vOBJd0F+CbwZdvfqoevlbSh7Wvq9NHr6vEFwL2H7r4x8Md6fONxjg/fZ4GklYC1gb+ObYftw4DDAObPn+8dd9yxg9+uW4d8+Tt85MJu/x0HbLlwRsRsEveqF+y4zJ857bTT6Pp/PVNjvuTA77WOOVOeT9PlOZqYiTnVcRNzbsbsK25iJuZ0jztTYn5x1zU6P8/rW59VTAUcDvza9keHbjoO2Kte3gv4ztDxPWtl0k0pxWjOrtNQb5a0Q4354jH3GcR6NvAj9z0kGhERERERMUv1OYL4SOBFwIWSzq/H3gZ8ADhG0j6U6aN7ANi+WNIxwCWUCqivsX1Hvd+rgC8CqwEn1C8oCeiRkq6gjBzu2ePvExERERERMav1liDa/ikwUUXRnSe4z0HAQeMcPwfYYpzjt1ETzIiIiIiIiGhnSqqYRkRERERExPSXBDEiIiIiIiKAJIgRERERERFRJUGMiIiIiIgIIAliREREREREVH1ucxHRm3nLsfH7AVsu7GSD+MSMiIiIiLkiI4gREREREREBJEGMiIiIiIiIKgliREREREREAEkQIyIiIiIiokqCGBEREREREUASxIiIiIiIiKiSIEZERERERASQBDEiIiIiIiKqJIgREREREREBJEGMiIiIiIiIKgliREREREREAEkQIyIiIiIiokqCGBEREREREUASxIiIiIiIiKiSIEZERERERASQBDEiIiIiIiKqJIgREREREREBJEGMiIiIiIiIKgliREREREREAEkQIyIiIiIiokqCGBEREREREUASxIiIiIiIiKiSIEZERERERASQBDEiIiIiIiKqJIgREREREREBJEGMiIiIiIiIKgliREREREREAEkQIyIiIiIiokqCGBEREREREUASxIiIiIiIiKh6SxAlfV7SdZIuGjq2rqSTJV1ev99t6La3SrpC0mWSdhk6vq2kC+ttn5CkenwVSV+rx8+SNK+v3yUiIiIiImIu6HME8YvArmOOHQicYnsz4JR6HUmbA3sCD673+bSkFet9DgX2BTarX4OY+wA32L4/8DHgg739JhEREREREXNAbwmi7dOBv445vDtwRL18BPD0oeNH277d9m+BK4DtJW0IrGX7DNsGvjTmPoNY3wB2HowuRkRERERExORN9RrEDWxfA1C/36Me3wj4w9DPLajHNqqXxx5f7D62FwI3AXfvreURERERERGznMrAXE/By7rA79reol6/0fY6Q7ffYPtukj4FnGH7qHr8cOD7wO+Bg20/vh5/NPBm27tJuhjYxfaCettvgO1tXz9OO/alTFNlgw022Pboo4/u7Xdu6rq/3sS1/+g25garMSNi9hU3MRNzusdNzLkZs6+4iTk3Y/YVNzETc7rHnSkxN117RdZcc81ug3Zgp512Otf2/PFuW2mK23KtpA1tX1Onj15Xjy8A7j30cxsDf6zHNx7n+PB9FkhaCVibJae0AmD7MOAwgPnz53vHHXfs5rfp0CFf/g4fubDbf8cBWy6cETH7ipuYiTnd4ybm3IzZV9zEnJsx+4qbmIk53ePOlJhf3HUNpmPusTRTPcX0OGCvenkv4DtDx/eslUk3pRSjObtOQ71Z0g51feGLx9xnEOvZwI/c53BoRERERETELNfbCKKkrwI7AutJWgC8C/gAcIykfSjTR/cAsH2xpGOAS4CFwGts31FDvYpSEXU14IT6BXA4cKSkKygjh3v29btERERERETMBb0liLafN8FNO0/w8wcBB41z/Bxgi3GO30ZNMCMiIiIiIqK9qZ5iGhEREREREdNUEsSIiIiIiIgAkiBGRERERERElQQxIiIiIiIigCSIERERERERUSVBjIiIiIiICCAJYkRERERERFRJECMiIiIiIgJIghgRERERERFVEsSIiIiIiIgAkiBGRERERERElQQxIiIiIiIigCSIERERERERUSVBjIiIiIiICCAJYkRERERERFRJECMiIiIiIgJIghgRERERERFVEsSIiIiIiIgAkiBGRERERERElQQxIiIiIiIigCSIERERERERUSVBjIiIiIiICCAJYkRERERERFRJECMiIiIiIgJIghgRERERERFVEsSIiIiIiIgAkiBGRERERERElQQxIiIiIiIigCSIERERERERUSVBjIiIiIiICCAJYkRERERERFRJECMiIiIiIgJIghgRERERERFVEsSIiIiIiIgAkiBGRERERERENeMTREm7SrpM0hWSDhx1eyIiIiIiImaqGZ0gSloR+BTwJGBz4HmSNh9tqyIiIiIiImam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GHWjZqMkiBERERERMd09H3gOcG392qMei45limlEREREREQAKVITERERERHTnKT1gZcD8xjKYWy/dFRtmq2SIEZERERExHT3HeAnwA/J2sNeZYppRERERERMa5LOt731qNsxF6RITURERERETHfflfTkUTdiLsgIYkRERERETGuSbgbWAG4H/gUIsO21RtqwWShrECMiIiIiYlqzfVdJ6wKbAauOuj2zWRLEiIiIiIiY1iS9DHg9sDFwPrAD8HNg5xE2a1bKGsSIiIiIiJjuXg9sB/zO9k7AQ4G/jLZJs1MSxIiIiIiImO5us30bgKRVbF8KPHDEbZqVMsU0IiIiIiKmuwWS1gG+DZws6QbgjyNt0SyVKqYRERERETFjSHossDZwou1/jro9s00SxIiIiIiIiACyBjEiIiIiIiKqJIgREREREREBJEGMiIjohaT9Ja3e1c9FRERMhaxBjIiI6IGkq4D5tpe6T9fy/lxERMRUyAhiRERES5LWkPQ9SRdIukjSu4B7AadKOrX+zKGSzpF0saT31GP7jfNzT5R0hqRfSvq6pDVH9XtFRMTckxHEiIiIliQ9C9jV9svr9bWBCxgaGZS0ru2/SloROAXYz/avhkcQJa0HfAt4ku2/S3oLsIrt947i94qIiLknI4gRERHtXQg8XtIHJT3a9k3j/MxzJP0SOA94MLD5OD+zQz3+M0nnA3sB9+mpzREREUtYadQNiIiImOls/5+kbYEnAwdLOmn4dkmbAm8CtrN9g6QvAquOE0rAybaf13ebIyIixpMRxIiIiJYk3Qu41fZRwH8D2wA3A3etP7IW8HfgJkkbAE8auvvwz50JPFLS/Wvc1SU9YAp+hYiICCAjiBEREV3YEviwpH8D/wJeBTwcOEHSNbZ3knQecDFwJfCzofseNubnXgJ8VdIq9fZ3AP83Vb9IRETMbSlSExEREREREUCmmEZERERERESVBDEiIiIiIiKAJIgRERERERFRJUGMiIiIiIgIIAliREREREREVEkQIyIiIiIiAkiCGBEREREREVUSxIiIiIiIiADg/wO5zu022/MHIwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1080x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,6))\n",
    "plt.xticks(rotation=90)\n",
    "df.state.hist()\n",
    "plt.xlabel('state')\n",
    "plt.ylabel('Frequencies')\n",
    "plt.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "23a4bfe8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Residential, Rural and other Areas    179014\n",
       "Industrial Area                        96091\n",
       "Residential and others                 86791\n",
       "Industrial Areas                       51747\n",
       "Sensitive Area                          8980\n",
       "Sensitive Areas                         5536\n",
       "RIRUO                                   1304\n",
       "Sensitive                                495\n",
       "Industrial                               233\n",
       "Residential                              158\n",
       "Name: type, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['type'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "8deffebe",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,6))\n",
    "plt.xticks(rotation=90)\n",
    "df.type.hist()\n",
    "plt.xlabel('type')\n",
    "plt.ylabel('Frequencies')\n",
    "plt.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "02a9a7d0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Maharashtra State Pollution Control Board                                 27857\n",
       "Uttar Pradesh State Pollution Control Board                               22686\n",
       "Andhra Pradesh State Pollution Control Board                              19139\n",
       "Himachal Pradesh State Environment Proection & Pollution Control Board    15287\n",
       "Punjab State Pollution Control Board                                      15232\n",
       "                                                                          ...  \n",
       "Arunachal Pradesh State Pollution Control Board                              90\n",
       "TNPC                                                                         82\n",
       "RPCB                                                                         63\n",
       "VRCE                                                                         61\n",
       "RJPB                                                                         53\n",
       "Name: agency, Length: 64, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['agency'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "f66511fd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(20,6))\n",
    "plt.xticks(rotation=90)\n",
    "df.agency.hist()\n",
    "plt.xlabel('agency')\n",
    "plt.ylabel('Frequencies')\n",
    "plt.plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "f2afec13",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='state', ylabel='so2'>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(20,10))\n",
    "plt.xticks(rotation=90)\n",
    "sns.barplot(x='state',y='so2', data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "c5859df9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 2160x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.rcParams['figure.figsize']=(30,10)\n",
    "df[['so2','state']].groupby(['state']).mean().sort_values(by='so2').plot.bar(color='green')\n",
    "plt.ylabel('so2')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "04d7b38c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='state', ylabel='pm10'>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(20,10))\n",
    "plt.xticks(rotation=90)\n",
    "sns.barplot(x='state',y='pm10', data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "618839d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:xlabel='state', ylabel='pm2_5'>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1440x720 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(20,10))\n",
    "plt.xticks(rotation=90)\n",
    "sns.barplot(x='state',y='pm2_5', data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "e10d02e8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "pm2_5                          237387\n",
       "agency                         149481\n",
       "stn_code                       144077\n",
       "pm10                            40222\n",
       "so2                             34646\n",
       "location_monitoring_station     27491\n",
       "no2                             16233\n",
       "type                             5393\n",
       "date                                7\n",
       "sampling_date                       3\n",
       "location                            3\n",
       "state                               0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nullvalues=df.isnull().sum().sort_values(ascending=False)\n",
    "nullvalues"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "361495c5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Total</th>\n",
       "      <th>Percnet</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>pm2_5</th>\n",
       "      <td>237387</td>\n",
       "      <td>54.478797</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>agency</th>\n",
       "      <td>149481</td>\n",
       "      <td>34.304933</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>stn_code</th>\n",
       "      <td>144077</td>\n",
       "      <td>33.064749</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pm10</th>\n",
       "      <td>40222</td>\n",
       "      <td>9.230692</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>so2</th>\n",
       "      <td>34646</td>\n",
       "      <td>7.951035</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>location_monitoring_station</th>\n",
       "      <td>27491</td>\n",
       "      <td>6.309009</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>no2</th>\n",
       "      <td>16233</td>\n",
       "      <td>3.725370</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>type</th>\n",
       "      <td>5393</td>\n",
       "      <td>1.237659</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>date</th>\n",
       "      <td>7</td>\n",
       "      <td>0.001606</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>sampling_date</th>\n",
       "      <td>3</td>\n",
       "      <td>0.000688</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>location</th>\n",
       "      <td>3</td>\n",
       "      <td>0.000688</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>state</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                              Total    Percnet\n",
       "pm2_5                        237387  54.478797\n",
       "agency                       149481  34.304933\n",
       "stn_code                     144077  33.064749\n",
       "pm10                          40222   9.230692\n",
       "so2                           34646   7.951035\n",
       "location_monitoring_station   27491   6.309009\n",
       "no2                           16233   3.725370\n",
       "type                           5393   1.237659\n",
       "date                              7   0.001606\n",
       "sampling_date                     3   0.000688\n",
       "location                          3   0.000688\n",
       "state                             0   0.000000"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "null_values_percentage=(df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)\n",
    "missing_data_with_percentage= pd.concat([nullvalues,null_values_percentage], axis=1, keys=['Total','Percnet'])\n",
    "missing_data_with_percentage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "b0d5e0dc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>stn_code</th>\n",
       "      <th>sampling_date</th>\n",
       "      <th>state</th>\n",
       "      <th>location</th>\n",
       "      <th>agency</th>\n",
       "      <th>type</th>\n",
       "      <th>so2</th>\n",
       "      <th>no2</th>\n",
       "      <th>pm10</th>\n",
       "      <th>pm2_5</th>\n",
       "      <th>location_monitoring_station</th>\n",
       "      <th>date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>150.0</td>\n",
       "      <td>February - M021990</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>4.8</td>\n",
       "      <td>17.4</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2/1/1990</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>151.0</td>\n",
       "      <td>February - M021990</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Industrial Area</td>\n",
       "      <td>3.1</td>\n",
       "      <td>7.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2/1/1990</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>152.0</td>\n",
       "      <td>February - M021990</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>6.2</td>\n",
       "      <td>28.5</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2/1/1990</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>150.0</td>\n",
       "      <td>March - M031990</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>6.3</td>\n",
       "      <td>14.7</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3/1/1990</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>151.0</td>\n",
       "      <td>March - M031990</td>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Industrial Area</td>\n",
       "      <td>4.7</td>\n",
       "      <td>7.5</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3/1/1990</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435737</th>\n",
       "      <td>SAMP</td>\n",
       "      <td>24-12-15</td>\n",
       "      <td>West Bengal</td>\n",
       "      <td>ULUBERIA</td>\n",
       "      <td>West Bengal State Pollution Control Board</td>\n",
       "      <td>RIRUO</td>\n",
       "      <td>22.0</td>\n",
       "      <td>50.0</td>\n",
       "      <td>143.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Inside Rampal Industries,ULUBERIA</td>\n",
       "      <td>12/24/2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435738</th>\n",
       "      <td>SAMP</td>\n",
       "      <td>29-12-15</td>\n",
       "      <td>West Bengal</td>\n",
       "      <td>ULUBERIA</td>\n",
       "      <td>West Bengal State Pollution Control Board</td>\n",
       "      <td>RIRUO</td>\n",
       "      <td>20.0</td>\n",
       "      <td>46.0</td>\n",
       "      <td>171.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Inside Rampal Industries,ULUBERIA</td>\n",
       "      <td>12/29/2015</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435739</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>andaman-and-nicobar-islands</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435740</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Lakshadweep</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435741</th>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Tripura</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>435742 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       stn_code       sampling_date                        state   location  \\\n",
       "0         150.0  February - M021990               Andhra Pradesh  Hyderabad   \n",
       "1         151.0  February - M021990               Andhra Pradesh  Hyderabad   \n",
       "2         152.0  February - M021990               Andhra Pradesh  Hyderabad   \n",
       "3         150.0     March - M031990               Andhra Pradesh  Hyderabad   \n",
       "4         151.0     March - M031990               Andhra Pradesh  Hyderabad   \n",
       "...         ...                 ...                          ...        ...   \n",
       "435737     SAMP            24-12-15                  West Bengal   ULUBERIA   \n",
       "435738     SAMP            29-12-15                  West Bengal   ULUBERIA   \n",
       "435739      NaN                 NaN  andaman-and-nicobar-islands        NaN   \n",
       "435740      NaN                 NaN                  Lakshadweep        NaN   \n",
       "435741      NaN                 NaN                      Tripura        NaN   \n",
       "\n",
       "                                           agency  \\\n",
       "0                                             NaN   \n",
       "1                                             NaN   \n",
       "2                                             NaN   \n",
       "3                                             NaN   \n",
       "4                                             NaN   \n",
       "...                                           ...   \n",
       "435737  West Bengal State Pollution Control Board   \n",
       "435738  West Bengal State Pollution Control Board   \n",
       "435739                                        NaN   \n",
       "435740                                        NaN   \n",
       "435741                                        NaN   \n",
       "\n",
       "                                      type   so2   no2   pm10  pm2_5  \\\n",
       "0       Residential, Rural and other Areas   4.8  17.4    NaN    NaN   \n",
       "1                          Industrial Area   3.1   7.0    NaN    NaN   \n",
       "2       Residential, Rural and other Areas   6.2  28.5    NaN    NaN   \n",
       "3       Residential, Rural and other Areas   6.3  14.7    NaN    NaN   \n",
       "4                          Industrial Area   4.7   7.5    NaN    NaN   \n",
       "...                                    ...   ...   ...    ...    ...   \n",
       "435737                               RIRUO  22.0  50.0  143.0    NaN   \n",
       "435738                               RIRUO  20.0  46.0  171.0    NaN   \n",
       "435739                                 NaN   NaN   NaN    NaN    NaN   \n",
       "435740                                 NaN   NaN   NaN    NaN    NaN   \n",
       "435741                                 NaN   NaN   NaN    NaN    NaN   \n",
       "\n",
       "              location_monitoring_station        date  \n",
       "0                                     NaN    2/1/1990  \n",
       "1                                     NaN    2/1/1990  \n",
       "2                                     NaN    2/1/1990  \n",
       "3                                     NaN    3/1/1990  \n",
       "4                                     NaN    3/1/1990  \n",
       "...                                   ...         ...  \n",
       "435737  Inside Rampal Industries,ULUBERIA  12/24/2015  \n",
       "435738  Inside Rampal Industries,ULUBERIA  12/29/2015  \n",
       "435739                                NaN         NaN  \n",
       "435740                                NaN         NaN  \n",
       "435741                                NaN         NaN  \n",
       "\n",
       "[435742 rows x 12 columns]"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "78207759",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(['agency'],axis=1, inplace=True)\n",
    "df.drop(['stn_code'],axis=1, inplace=True)\n",
    "df.drop(['date'],axis=1, inplace=True)\n",
    "df.drop(['sampling_date'],axis=1, inplace=True)\n",
    "df.drop(['location_monitoring_station'],axis=1, inplace=True)\n",
    "#dropping unnecessary columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "bd5023fa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "state            0\n",
       "location         3\n",
       "type          5393\n",
       "so2          34646\n",
       "no2          16233\n",
       "pm10         40222\n",
       "pm2_5       237387\n",
       "dtype: int64"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "28a61778",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>state</th>\n",
       "      <th>location</th>\n",
       "      <th>type</th>\n",
       "      <th>so2</th>\n",
       "      <th>no2</th>\n",
       "      <th>pm10</th>\n",
       "      <th>pm2_5</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>4.8</td>\n",
       "      <td>17.4</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Industrial Area</td>\n",
       "      <td>3.1</td>\n",
       "      <td>7.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>6.2</td>\n",
       "      <td>28.5</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>6.3</td>\n",
       "      <td>14.7</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Industrial Area</td>\n",
       "      <td>4.7</td>\n",
       "      <td>7.5</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435737</th>\n",
       "      <td>West Bengal</td>\n",
       "      <td>ULUBERIA</td>\n",
       "      <td>RIRUO</td>\n",
       "      <td>22.0</td>\n",
       "      <td>50.0</td>\n",
       "      <td>143.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435738</th>\n",
       "      <td>West Bengal</td>\n",
       "      <td>ULUBERIA</td>\n",
       "      <td>RIRUO</td>\n",
       "      <td>20.0</td>\n",
       "      <td>46.0</td>\n",
       "      <td>171.0</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435739</th>\n",
       "      <td>andaman-and-nicobar-islands</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435740</th>\n",
       "      <td>Lakshadweep</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435741</th>\n",
       "      <td>Tripura</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>435742 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                              state   location  \\\n",
       "0                    Andhra Pradesh  Hyderabad   \n",
       "1                    Andhra Pradesh  Hyderabad   \n",
       "2                    Andhra Pradesh  Hyderabad   \n",
       "3                    Andhra Pradesh  Hyderabad   \n",
       "4                    Andhra Pradesh  Hyderabad   \n",
       "...                             ...        ...   \n",
       "435737                  West Bengal   ULUBERIA   \n",
       "435738                  West Bengal   ULUBERIA   \n",
       "435739  andaman-and-nicobar-islands        NaN   \n",
       "435740                  Lakshadweep        NaN   \n",
       "435741                      Tripura        NaN   \n",
       "\n",
       "                                      type   so2   no2   pm10  pm2_5  \n",
       "0       Residential, Rural and other Areas   4.8  17.4    NaN    NaN  \n",
       "1                          Industrial Area   3.1   7.0    NaN    NaN  \n",
       "2       Residential, Rural and other Areas   6.2  28.5    NaN    NaN  \n",
       "3       Residential, Rural and other Areas   6.3  14.7    NaN    NaN  \n",
       "4                          Industrial Area   4.7   7.5    NaN    NaN  \n",
       "...                                    ...   ...   ...    ...    ...  \n",
       "435737                               RIRUO  22.0  50.0  143.0    NaN  \n",
       "435738                               RIRUO  20.0  46.0  171.0    NaN  \n",
       "435739                                 NaN   NaN   NaN    NaN    NaN  \n",
       "435740                                 NaN   NaN   NaN    NaN    NaN  \n",
       "435741                                 NaN   NaN   NaN    NaN    NaN  \n",
       "\n",
       "[435742 rows x 7 columns]"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "73a459d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#null value imputation for categorical data\n",
    "df['location']=df['location'].fillna(df['location'].mode()[0])\n",
    "df['type']=df['type'].fillna(df['type'].mode()[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "d27a4e73",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.fillna(0, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "9cb606e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "state       0\n",
       "location    0\n",
       "type        0\n",
       "so2         0\n",
       "no2         0\n",
       "pm10        0\n",
       "pm2_5       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "1972e972",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>state</th>\n",
       "      <th>location</th>\n",
       "      <th>type</th>\n",
       "      <th>so2</th>\n",
       "      <th>no2</th>\n",
       "      <th>pm10</th>\n",
       "      <th>pm2_5</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>4.8</td>\n",
       "      <td>17.4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Industrial Area</td>\n",
       "      <td>3.1</td>\n",
       "      <td>7.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>6.2</td>\n",
       "      <td>28.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>6.3</td>\n",
       "      <td>14.7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Industrial Area</td>\n",
       "      <td>4.7</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435737</th>\n",
       "      <td>West Bengal</td>\n",
       "      <td>ULUBERIA</td>\n",
       "      <td>RIRUO</td>\n",
       "      <td>22.0</td>\n",
       "      <td>50.0</td>\n",
       "      <td>143.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435738</th>\n",
       "      <td>West Bengal</td>\n",
       "      <td>ULUBERIA</td>\n",
       "      <td>RIRUO</td>\n",
       "      <td>20.0</td>\n",
       "      <td>46.0</td>\n",
       "      <td>171.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435739</th>\n",
       "      <td>andaman-and-nicobar-islands</td>\n",
       "      <td>Guwahati</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435740</th>\n",
       "      <td>Lakshadweep</td>\n",
       "      <td>Guwahati</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>435741</th>\n",
       "      <td>Tripura</td>\n",
       "      <td>Guwahati</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>435742 rows × 7 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                              state   location  \\\n",
       "0                    Andhra Pradesh  Hyderabad   \n",
       "1                    Andhra Pradesh  Hyderabad   \n",
       "2                    Andhra Pradesh  Hyderabad   \n",
       "3                    Andhra Pradesh  Hyderabad   \n",
       "4                    Andhra Pradesh  Hyderabad   \n",
       "...                             ...        ...   \n",
       "435737                  West Bengal   ULUBERIA   \n",
       "435738                  West Bengal   ULUBERIA   \n",
       "435739  andaman-and-nicobar-islands   Guwahati   \n",
       "435740                  Lakshadweep   Guwahati   \n",
       "435741                      Tripura   Guwahati   \n",
       "\n",
       "                                      type   so2   no2   pm10  pm2_5  \n",
       "0       Residential, Rural and other Areas   4.8  17.4    0.0    0.0  \n",
       "1                          Industrial Area   3.1   7.0    0.0    0.0  \n",
       "2       Residential, Rural and other Areas   6.2  28.5    0.0    0.0  \n",
       "3       Residential, Rural and other Areas   6.3  14.7    0.0    0.0  \n",
       "4                          Industrial Area   4.7   7.5    0.0    0.0  \n",
       "...                                    ...   ...   ...    ...    ...  \n",
       "435737                               RIRUO  22.0  50.0  143.0    0.0  \n",
       "435738                               RIRUO  20.0  46.0  171.0    0.0  \n",
       "435739  Residential, Rural and other Areas   0.0   0.0    0.0    0.0  \n",
       "435740  Residential, Rural and other Areas   0.0   0.0    0.0    0.0  \n",
       "435741  Residential, Rural and other Areas   0.0   0.0    0.0    0.0  \n",
       "\n",
       "[435742 rows x 7 columns]"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e209c5a5",
   "metadata": {},
   "source": [
    "# Calculate Air Quality Index for So2 based on Formula"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0ecf6284",
   "metadata": {},
   "source": [
    "function to calculate so2 individual pollutant index(si)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "00543520",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>so2</th>\n",
       "      <th>Soi</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.8</td>\n",
       "      <td>6.000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3.1</td>\n",
       "      <td>3.875</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6.2</td>\n",
       "      <td>7.750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6.3</td>\n",
       "      <td>7.875</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4.7</td>\n",
       "      <td>5.875</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   so2    Soi\n",
       "0  4.8  6.000\n",
       "1  3.1  3.875\n",
       "2  6.2  7.750\n",
       "3  6.3  7.875\n",
       "4  4.7  5.875"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def cal_soi(so2):\n",
    "    si=0\n",
    "    if (so2<=40):\n",
    "        si=so2*(50/40)\n",
    "    elif (so2>40 and so2<=80):\n",
    "        si=50 + (so2 - 40) * (50 / 40)\n",
    "    elif (so2>80 and so2<=380):\n",
    "        si=100 + (so2 - 80) * (100 / 300)\n",
    "    elif (so2>380 and so2<=800):\n",
    "        si=200 + (so2 - 380) * (100 / 420)\n",
    "    elif (so2>800 and so2<=1600):\n",
    "        si=300 + (so2 - 800) * (100 / 800)\n",
    "    elif (so2>1600):\n",
    "        si=400 + (so2 - 1600) * (100 / 800)\n",
    "    return si\n",
    "df['Soi']=df['so2'].apply(cal_soi)\n",
    "data= df[['so2','Soi']]\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6ca12750",
   "metadata": {},
   "source": [
    "Function to calculate no2 individual pollutant index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "becd76d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>no2</th>\n",
       "      <th>Noi</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>17.4</td>\n",
       "      <td>21.750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>7.0</td>\n",
       "      <td>8.750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>28.5</td>\n",
       "      <td>35.625</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>14.7</td>\n",
       "      <td>18.375</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7.5</td>\n",
       "      <td>9.375</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    no2     Noi\n",
       "0  17.4  21.750\n",
       "1   7.0   8.750\n",
       "2  28.5  35.625\n",
       "3  14.7  18.375\n",
       "4   7.5   9.375"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def cal_noi(no2):\n",
    "    ni=0\n",
    "    if (no2<=40):\n",
    "        ni=no2*(50/40)\n",
    "    elif (no2>40 and no2<=80):\n",
    "        ni=50 + (no2 - 40) * (50 / 40)\n",
    "    elif (no2>80 and no2<=180):\n",
    "        ni=100 + (no2 - 80) * (100 / 100)\n",
    "    elif (no2>180 and no2<=280):\n",
    "        ni=200 + (no2 - 180) * (100 / 100)\n",
    "    elif (no2>280 and no2<=400):\n",
    "        ni=300 + (no2 - 280) * (100 / 120)\n",
    "    else:\n",
    "        ni=400 + (no2 - 400) * (100 / 120)\n",
    "    return ni\n",
    "df['Noi']=df['no2'].apply(cal_noi)\n",
    "data= df[['no2','Noi']]\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1363c712",
   "metadata": {},
   "source": [
    "Function to calculate pm10 individual pollutant index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "8a66cde8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>pm10</th>\n",
       "      <th>pm10i</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   pm10  pm10i\n",
       "0   0.0    0.0\n",
       "1   0.0    0.0\n",
       "2   0.0    0.0\n",
       "3   0.0    0.0\n",
       "4   0.0    0.0"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def cal_pm10i(pm10):\n",
    "    pm10i=0\n",
    "    if (pm10i<=30):\n",
    "        pm10i=pm10i*(50/30)\n",
    "    elif (pm10i>30 and pm10i<=60):\n",
    "        pm10i=50 + (pm10i - 30) * (50 / 30)\n",
    "    elif (pm10i>60 and pm10i<=90):\n",
    "        pm10i=100 + (pm10i - 60) * (100 / 30)\n",
    "    elif (pm10i>90 and pm10i<=120):\n",
    "        pm10i=200 + (pm10i - 90) * (100 / 30)\n",
    "    elif (pm10i>120 and pm10i<=250):\n",
    "        pm10i=300 + (pm10i - 120) * (100 / 130)\n",
    "    else:\n",
    "        pm10i=400 + (pm10i - 250) * (100 / 130)\n",
    "    return pm10i\n",
    "df['pm10i']=df['pm10'].apply(cal_pm10i)\n",
    "data= df[['pm10','pm10i']]\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "871ce87c",
   "metadata": {},
   "source": [
    "Function to calculate pm2_5 individual pollutant index(spi)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "ed066ad6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>pm2_5</th>\n",
       "      <th>pm2_5i</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   pm2_5  pm2_5i\n",
       "0    0.0     0.0\n",
       "1    0.0     0.0\n",
       "2    0.0     0.0\n",
       "3    0.0     0.0\n",
       "4    0.0     0.0"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def cal_pm2_5i(pm2_5):\n",
    "    pm2_5i=0\n",
    "    if (pm2_5<=30):\n",
    "        pm2_5i=pm2_5*(50/50)\n",
    "    elif (pm2_5>50 and pm2_5<=100):\n",
    "        pm2_5i=50 + (pm2_5 - 50) * (50 / 30)\n",
    "    elif (pm2_5>100 and pm2_5<=250):\n",
    "        pm2_5i=100 + (pm2_5 - 100) * (100 / 150)\n",
    "    elif (pm2_5>250 and pm2_5<=350):\n",
    "        pm2_5i=200 + (pm2_5 - 250) * (100 / 100)\n",
    "    elif (pm2_5>350 and pm2_5<=430):\n",
    "        pm2_5i=300 + (pm2_5 - 350) * (100 / 80)\n",
    "    else:\n",
    "        pm2_5i=400 + (pm2_5 - 430) * (100 / 430)\n",
    "    return pm2_5i\n",
    "df['pm2_5i']=df['pm2_5'].apply(cal_pm2_5i)\n",
    "data= df[['pm2_5','pm2_5i']]\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b4d8e575",
   "metadata": {},
   "source": [
    "Function to calculate Air Quality Index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "c67afb83",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>state</th>\n",
       "      <th>Soi</th>\n",
       "      <th>Noi</th>\n",
       "      <th>pm10i</th>\n",
       "      <th>pm2_5i</th>\n",
       "      <th>AQI</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>6.000</td>\n",
       "      <td>21.750</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>21.750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>3.875</td>\n",
       "      <td>8.750</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.750</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>7.750</td>\n",
       "      <td>35.625</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>35.625</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>7.875</td>\n",
       "      <td>18.375</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>18.375</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>5.875</td>\n",
       "      <td>9.375</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.375</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            state    Soi     Noi  pm10i  pm2_5i     AQI\n",
       "0  Andhra Pradesh  6.000  21.750    0.0     0.0  21.750\n",
       "1  Andhra Pradesh  3.875   8.750    0.0     0.0   8.750\n",
       "2  Andhra Pradesh  7.750  35.625    0.0     0.0  35.625\n",
       "3  Andhra Pradesh  7.875  18.375    0.0     0.0  18.375\n",
       "4  Andhra Pradesh  5.875   9.375    0.0     0.0   9.375"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def cal_aqi(si,ni,pm10i,pm2_5i):\n",
    "    aqi=0\n",
    "    if(si>ni and si>pm10i and si>pm2_5i):\n",
    "        aqi=si\n",
    "    if(ni>si and ni>pm10i and ni>pm2_5i):\n",
    "        aqi=ni\n",
    "    if(pm10i>si and pm10i>ni and pm10i>pm2_5i):\n",
    "        aqi=pm10i\n",
    "    if(pm2_5i>si and pm2_5i>ni and pm2_5i>pm10i):\n",
    "        aqi=pm2_5i\n",
    "    return aqi\n",
    "df['AQI']=df.apply(lambda x:cal_aqi(x['Soi'],x['Noi'],x['pm10i'],x['pm2_5i']),axis=1)\n",
    "data=df[['state','Soi','Noi','pm10i','pm2_5i','AQI']]\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "53c9555a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>state</th>\n",
       "      <th>location</th>\n",
       "      <th>type</th>\n",
       "      <th>so2</th>\n",
       "      <th>no2</th>\n",
       "      <th>pm10</th>\n",
       "      <th>pm2_5</th>\n",
       "      <th>Soi</th>\n",
       "      <th>Noi</th>\n",
       "      <th>pm10i</th>\n",
       "      <th>pm2_5i</th>\n",
       "      <th>AQI</th>\n",
       "      <th>AQI_Range</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>4.8</td>\n",
       "      <td>17.4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>6.000</td>\n",
       "      <td>21.750</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>21.750</td>\n",
       "      <td>Good</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Industrial Area</td>\n",
       "      <td>3.1</td>\n",
       "      <td>7.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.875</td>\n",
       "      <td>8.750</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8.750</td>\n",
       "      <td>Good</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>6.2</td>\n",
       "      <td>28.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.750</td>\n",
       "      <td>35.625</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>35.625</td>\n",
       "      <td>Good</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Residential, Rural and other Areas</td>\n",
       "      <td>6.3</td>\n",
       "      <td>14.7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.875</td>\n",
       "      <td>18.375</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>18.375</td>\n",
       "      <td>Good</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Andhra Pradesh</td>\n",
       "      <td>Hyderabad</td>\n",
       "      <td>Industrial Area</td>\n",
       "      <td>4.7</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5.875</td>\n",
       "      <td>9.375</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.375</td>\n",
       "      <td>Good</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            state   location                                type  so2   no2  \\\n",
       "0  Andhra Pradesh  Hyderabad  Residential, Rural and other Areas  4.8  17.4   \n",
       "1  Andhra Pradesh  Hyderabad                     Industrial Area  3.1   7.0   \n",
       "2  Andhra Pradesh  Hyderabad  Residential, Rural and other Areas  6.2  28.5   \n",
       "3  Andhra Pradesh  Hyderabad  Residential, Rural and other Areas  6.3  14.7   \n",
       "4  Andhra Pradesh  Hyderabad                     Industrial Area  4.7   7.5   \n",
       "\n",
       "   pm10  pm2_5    Soi     Noi  pm10i  pm2_5i     AQI AQI_Range  \n",
       "0   0.0    0.0  6.000  21.750    0.0     0.0  21.750      Good  \n",
       "1   0.0    0.0  3.875   8.750    0.0     0.0   8.750      Good  \n",
       "2   0.0    0.0  7.750  35.625    0.0     0.0  35.625      Good  \n",
       "3   0.0    0.0  7.875  18.375    0.0     0.0  18.375      Good  \n",
       "4   0.0    0.0  5.875   9.375    0.0     0.0   9.375      Good  "
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#using threshold value to classify a particular value as good, moderate, poor, unhealthy, very unhealthy, hazardous\n",
    "def AQI_Range(x):\n",
    "    if x<=50:\n",
    "        return \"Good\"\n",
    "    elif x>50 and x<=100:\n",
    "        return \"Moderate\"\n",
    "    elif x>100 and x<=200:\n",
    "        return \"Poor\"\n",
    "    elif x>200 and x<=300:\n",
    "        return \"Unhealthy\"\n",
    "    elif x>300 and x<=400:\n",
    "        return \"Very unhealthy\"\n",
    "    elif x>400:\n",
    "        return \"Hazardous\"\n",
    "df['AQI_Range']=df['AQI'].apply(AQI_Range)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "9a7821b9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Good              212922\n",
       "Poor              106335\n",
       "Moderate           43267\n",
       "Unhealthy          31733\n",
       "Very unhealthy     22785\n",
       "Hazardous          18700\n",
       "Name: AQI_Range, dtype: int64"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['AQI_Range'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "680aeb71",
   "metadata": {},
   "source": [
    "Splitting the dataset into Dependent and Independent columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "a5f11400",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>so2</th>\n",
       "      <th>no2</th>\n",
       "      <th>pm10</th>\n",
       "      <th>pm2_5</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.8</td>\n",
       "      <td>17.4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>3.1</td>\n",
       "      <td>7.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>6.2</td>\n",
       "      <td>28.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>6.3</td>\n",
       "      <td>14.7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4.7</td>\n",
       "      <td>7.5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   so2   no2  pm10  pm2_5\n",
       "0  4.8  17.4   0.0    0.0\n",
       "1  3.1   7.0   0.0    0.0\n",
       "2  6.2  28.5   0.0    0.0\n",
       "3  6.3  14.7   0.0    0.0\n",
       "4  4.7   7.5   0.0    0.0"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X=df[['so2','no2','pm10','pm2_5']]\n",
    "Y=df['AQI']\n",
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "e47209c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    21.750\n",
       "1     8.750\n",
       "2    35.625\n",
       "3    18.375\n",
       "4     9.375\n",
       "Name: AQI, dtype: float64"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "eb6766a6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(348593, 4) (87149, 4) (348593,) (87149,)\n"
     ]
    }
   ],
   "source": [
    "#splitting the data into training and testing data\n",
    "x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=70)\n",
    "print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a2a20daa",
   "metadata": {},
   "source": [
    "### Random Forest Regressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "fd6ad101",
   "metadata": {},
   "outputs": [],
   "source": [
    "RF=RandomForestRegressor().fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "07019724",
   "metadata": {},
   "outputs": [],
   "source": [
    "#predicting on train\n",
    "train_preds1=RF.predict(x_train)\n",
    "#predicting on test\n",
    "test_preds1=RF.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "fbe16e82",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE TrainingData =  0.517901661374921\n",
      "RMSE TestData =  1.2165021556586042\n",
      "--------------------------------------------------\n",
      "RSquared value on train: 0.9999791356727521\n",
      "RSquared value on test: 0.999884449650663\n"
     ]
    }
   ],
   "source": [
    "RMSE_train=(np.sqrt(metrics.mean_squared_error(y_train,train_preds1)))\n",
    "RMSE_test=(np.sqrt(metrics.mean_squared_error(y_test,test_preds1)))\n",
    "print(\"RMSE TrainingData = \",str(RMSE_train))\n",
    "print(\"RMSE TestData = \",str(RMSE_test))\n",
    "print(\"-\"*50)\n",
    "print('RSquared value on train:',RF.score(x_train, y_train))\n",
    "print('RSquared value on test:',RF.score(x_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "a844b280",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([21.75])"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "RF.predict([[4.8,17.4,0,0]])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "923f87e9",
   "metadata": {},
   "source": [
    "### Classification Algorithm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "2cb0729e",
   "metadata": {},
   "outputs": [],
   "source": [
    "X2=df[['so2','no2','pm10','pm2_5']]\n",
    "Y2=df['AQI_Range']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "0753c0d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#splitting the data into training and testing data\n",
    "x_train2,x_test2,y_train2,y_test2=train_test_split(X2,Y2,test_size=0.33,random_state=70)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0647eca2",
   "metadata": {},
   "source": [
    "### K-Nearest Neighbors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "05671f75",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model accuracy on train is: 0.9959478946521115\n",
      "Model accuracy on test is: 0.9926144859000661\n",
      "--------------------------------------------------\n",
      "KappaScore is: 0.9891503866912992\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "#fit the model on train data\n",
    "knn=KNeighborsClassifier().fit(x_train2,y_train2)\n",
    "#prediction on train\n",
    "train_preds5=knn.predict(x_train2)\n",
    "#accuracy on train\n",
    "print(\"Model accuracy on train is:\", accuracy_score(y_train2,train_preds5))\n",
    "\n",
    "#prediction on test\n",
    "test_preds5=knn.predict(x_test2)\n",
    "#accuracy on test\n",
    "print(\"Model accuracy on test is:\", accuracy_score(y_test2,test_preds5))\n",
    "print(\"-\"*50)\n",
    "\n",
    "#kappa score\n",
    "print(\"KappaScore is:\", metrics.cohen_kappa_score(y_test2,test_preds5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "115330b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Good'], dtype=object)"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knn.predict([[4.8,17.4,0,0]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "965047bd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([21.75])"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "RF.predict([[4.8,17.4,0,0]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "4c0a2c4b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "# open a file, where you ant to store the data\n",
    "file = open('Algorithm.pkl', 'wb')\n",
    "\n",
    "# dump information to that file\n",
    "pickle.dump(RF, file)\n",
    "pickle.dump(knn, file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "7fe4d722",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AQI is :  [21.75]\n",
      "AQI is Classified as:  ['Good']\n"
     ]
    }
   ],
   "source": [
    "loaded_model=pickle.load(open('Algorithm.pkl','rb'))\n",
    "input_data=[4.8,17.4,0,0]\n",
    "print(\"AQI is : \",RF.predict([input_data]))\n",
    "print(\"AQI is Classified as: \",knn.predict([input_data]))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {},
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
