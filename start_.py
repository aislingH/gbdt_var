from gbdt_var import *
from sklearn.ensemble import GradientBoostingClassifier


def get_quick_ks(y_pred, y_true):
    return ks_2samp(y_pred[y_true == 1], y_pred[y_true != 1]).statistic


df_ = pd.read_csv(
    "/Users/aisling/Documents/lexin/G-projects-lx/Behav_matrix/data/201907/au_bscore_var_1907.csv")
train_cols = [i for i in df_.columns if '_' in i and 'prob' not in i and i not in [
    'etl_date', 'drv_mth'] and df_[i].dtype != object]
print(train_cols)

model = GradientBoostingClassifier(max_depth=4,
                                   min_samples_leaf=0.01, n_estimators=10, random_state=666)

# p1 = model.predict_proba()
model.fit(df_[train_cols], df_['bad'])

gbdt_path_var = get_gbdt_path_var(
    df_[train_cols], model, y=None, one_hot=False)
# get_head_rule(df_[train_cols], df_['bad'], head=5, cover=0.02)

rule_df = get_rule_df(gbdt_path_var, df_['bad'])
print(rule_df)

intercept, coef, cols = get_lr_model(
    gbdt_path_var, df_['bad'], C=0.005, random_state=1234)
get_lr_proba(intercept, coef, cols, gbdt_path_var, df_['bad'])
# AUC: 0.8212  KS: 0.483
