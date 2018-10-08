
        self.ML_df.to_csv('data.csv')
        self.X_train.to_csv('X_train.csv')
        self.y_train.to_csv('y_train.csv')
        self.weights_train.to_csv('weights_train.csv')

        self.X_test.to_csv('X_test.csv')
        self.y_test.to_csv('y_test.csv')
        self.weights_test.to_csv('weights_test.csv')


if cls.model_name_short == "XGBR":
    cls.hyper_parameters = {  
            "n_estimators": st.randint(3, 40),
            "max_depth": st.randint(3, 40),
            "learning_rate": st.uniform(0.05, 0.4),
            "colsample_bytree": one_to_left,
            "subsample": one_to_left,
            "gamma": st.uniform(0, 10),
            'reg_alpha': from_zero_positive,
            "min_child_weight": from_zero_positive,
        }