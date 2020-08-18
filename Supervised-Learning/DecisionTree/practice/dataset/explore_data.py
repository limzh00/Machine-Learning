import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

home_data_path = "./melb_data.csv"
def main():
    home_data = pd.read_csv(home_data_path)
    home_data = home_data.dropna(axis = 0)
    # print(home_data.describe())
    features = ['Rooms','Bathroom']
    X = home_data[features]
    y = home_data.Price
    clf = DecisionTreeRegressor()
    clf.fit(X,y)
    y_pre = clf.predict(X.head())
    print(y_pre, y.head())
if __name__ == '__main__':
    main()