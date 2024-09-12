import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('future.no_silent_downcasting', True)

students = pd.read_csv('data.csv', sep=';')
students.head()

students['Target'] = students['Target'].map({'Graduate':0, 'Dropout':1})
students['Unemployment rate'] = students['Unemployment rate']/100
students['Inflation rate'] = students['Inflation rate']/100

numerical_features = [
                        'GDP',
                        'Gender',
                        'Displaced',
                        'International',
                        'Scholarship holder',
                        'Curricular units 1st sem (credited)',
                        'Curricular units 1st sem (enrolled)',
                        'Curricular units 1st sem (evaluations)',
                        'Curricular units 1st sem (approved)',
                        'Curricular units 1st sem (grade)',
                        'Curricular units 1st sem (without evaluations)',
                        'Curricular units 2nd sem (credited)',
                        'Curricular units 2nd sem (enrolled)',
                        'Curricular units 2nd sem (evaluations)',
                        'Curricular units 2nd sem (approved)',
                        'Curricular units 2nd sem (grade)',
                        'Curricular units 2nd sem (without evaluations)',
                        'Age at enrollment',
                        'Tuition fees up to date',
                        'Debtor',
                        'Previous qualification (grade)',
                        'Educational special needs',
                        'Admission grade',
                        'Daytime/evening attendance	'
                      ]

rate_features = ['Unemployment rate', 'Inflation rate']

ordinal_features = ['Application order']

categorical_features = [
                        'Marital status',
                        'Application mode',
                        'Course',
                        'Previous qualification',
                        'Nacionality',
                        "Mother's qualification",
                        "Father's qualification",
                        "Mother's occupation",
                        "Father's occupation"
]

for col in categorical_features:
    students[col] = students[col].astype('category')
    
for col in ordinal_features:
    students[col] = students[col].astype('category')

target = 'Target'

X = students.drop(target, axis=1)
y = students[target]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder, OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

rate_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())
])

ordinal_transformer = Pipeline(steps=[
    ('ord_enc', OrdinalEncoder())
])

categorical_transformer = Pipeline(steps=[
    ('targ_enc', TargetEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
       ('num', numerical_transformer, numerical_features),
       ('rat', rate_transformer, rate_features),
       ('ord', ordinal_transformer, ordinal_features),
       ('cat', categorical_transformer, categorical_features) 
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=50, random_state=42))  
])


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
print(f"ROC/AUC: {roc_auc_score(y_test, y_pred):.2f}")


print('Matriz de Confusão:\n', confusion_matrix(y_test, y_pred))

classifier = pipeline.named_steps['classifier']

feature_importances = classifier.feature_importances_

all_features = numerical_features + rate_features + ordinal_features + categorical_features

feature_importance_df = pd.DataFrame({
    'features':all_features,
    'importance':feature_importances
    
})

feature_importance_df.sort_values(by='importance', ascending=False)

feature_importances = {}

# Obtém a lista de nomes das colunas (features) do conjunto de teste
features = X_test.columns.tolist()

# Calcula a acurácia do modelo com as previsões originais (baseline)
baseline_metric = accuracy_score(y_test, y_pred)

# Loop sobre cada feature no conjunto de teste
for feature in features:
    
    # Faz uma cópia do conjunto de teste para não modificar o original
    X_test_copy = X_test.copy()
    
    # Embaralha aleatoriamente os valores da feature atual, removendo qualquer correlação com o rótulo
    X_test_copy[feature] = np.random.permutation(X_test_copy[feature])
    
    # Faz novas previsões usando o conjunto de teste com a feature embaralhada
    y_pred_permuted = pipeline.predict(X_test_copy)
    
    # Calcula a acurácia do modelo com a feature embaralhada
    permuted_metric = accuracy_score(y_test, y_pred_permuted)
    
    # Calcula a diferença entre a acurácia original e a acurácia com a feature embaralhada
    # Quanto maior a diferença, mais importante é a feature para o modelo
    feature_importances[feature] = baseline_metric - permuted_metric


pd.Series(feature_importances).sort_values(ascending=False)



