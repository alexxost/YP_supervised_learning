#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Вводная-к-этому-проекту" data-toc-modified-id="Вводная-к-этому-проекту-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Вводная к этому проекту</a></span></li><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Подготовка данных</a></span></li><li><span><a href="#Исследование-задачи" data-toc-modified-id="Исследование-задачи-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Исследование задачи</a></span></li><li><span><a href="#Борьба-с-дисбалансом" data-toc-modified-id="Борьба-с-дисбалансом-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Борьба с дисбалансом</a></span></li><li><span><a href="#Сравнение-эффективности-методов-на-сбалансированном-микро-сете" data-toc-modified-id="Сравнение-эффективности-методов-на-сбалансированном-микро-сете-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Сравнение эффективности методов на сбалансированном микро-сете</a></span></li><li><span><a href="#We-need-to-go-smaller" data-toc-modified-id="We-need-to-go-smaller-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>We need to go smaller</a></span></li><li><span><a href="#Feature-importance" data-toc-modified-id="Feature-importance-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Feature importance</a></span></li><li><span><a href="#Общий-вывод" data-toc-modified-id="Общий-вывод-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Общий вывод</a></span></li><li><span><a href="#Чек-лист-готовности-проекта" data-toc-modified-id="Чек-лист-готовности-проекта-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Чек-лист готовности проекта</a></span></li></ul></div>

# # Отток клиентов

# Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
# 
# Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Вам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 
# 
# Постройте модель с предельно большим значением *F1*-меры. Чтобы сдать проект успешно, нужно довести метрику до 0.59. Проверьте *F1*-меру на тестовой выборке самостоятельно.
# 
# Дополнительно измеряйте *AUC-ROC*, сравнивайте её значение с *F1*-мерой.
# 
# Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

# ## Вводная к этому проекту

# Я решил принципиально сосредоточиться на модели RF, потому что мне поставили на работе задачу, непосредственно с ними связанную: мне, с их помощью, надо выявить feature importance для маленького дата-сета. Для этого я планирую использовать методы SHAP и LIME. К сожалению, чтобы не затягивать с и без того отстающей сдачей, и не возиться с совместимостью библиотек, в этом проекте я их использовать не смогу. Так что просто посмотрю, не меняется ли значимость с уменьшением дата-сета и использовании самого базового метода.

# ## Подготовка данных

# In[58]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10,7)})

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.inspection import permutation_importance


# In[2]:


based = pd.read_csv('/datasets/Churn.csv')
display(based.head(10))
display(based.info())


# Персональные данные здесь ни к чему: модель они будут только сбивать, в попытках понять как уход из банка зависит от вашей фамилии. Устраняем их без доли сожаления, лишь удостоверившись, впрочем, что число клиентов в БД не вырожденное. 

# In[3]:


print(len(based['CustomerId'].unique())==based.shape[0])
based = based.drop(['RowNumber','CustomerId','Surname'],axis=1)
display(based.tail(10))


# Здесь такой тонкий момент, что начав разбираться как заполнять пропуски всякими фэнси-эппроучами с proximity-matrix и иными приблудами я слегка заплутал. В итоге, даже вместо sklearn.impute.SimpleImputer, просто поставил вместо пропусков медианные значения, так как столбец такой только один, хоть и с ~10% пропусков.

# In[4]:


based['Tenure'] = based['Tenure'].fillna(based['Tenure'].median())
display(based.tail(10))


# In[5]:


print(based['Geography'].unique())
print(based['Gender'].unique())


# Поскольку категориальных совсем чуть-чуть, можно применять OHE без ужаса раздутия таблицы в лавкрафтовское чудовище. 

# In[6]:


based = pd.get_dummies(data=based,columns=['Geography','Gender'])
display(based.head(10))


# In[7]:


y = based['Exited']
X = based.drop('Exited',axis=1)


# Для Random Forest нет нужды нормализировать данные. Также, для RF деление идет на трейн-тест, без внешней валидационной выборки. Подбор гиперпараметров планируется вести двумя методами. Первый - без bootstrap, через RandomizedSearchCV, со внутренней кросс-валидацией, а второй через GridSearch, при использовании bootstrap. В этом случае метрикой является out-of-bag error. Рассмотрение обоих позволяет оценить bias-variance tradeoff для модели.

# In[8]:


tree_X_train, tree_X_test, tree_y_train, tree_y_test = train_test_split(X,y, test_size=0.1, random_state=42)


# In[9]:


tree = (tree_X_train, tree_X_test,tree_y_train, tree_y_test)


# Очевидно, что классы сильно несбалансированы: число ушедших клиентов ~20%. Посмотрим точнее.

# In[10]:


print(f'Class disbalance is: \n{y.value_counts(normalize=True)}')


# ## Исследование задачи

# **Зададим все рабочие функции**

# Execute прогоняет тест для модели и отображает требуемые выборки.

# In[11]:


def execute(model,data,talk,draw):
    
    model_upd = model.fit(data[0],data[2])
    result = model_upd.predict(data[1])
    f1 = f1_score(data[3], result)
    yeerokk = roc_auc_score(data[3], model_upd.predict_proba(data[1])[:, 1])
    
    if talk == True:
        print(f'For selected model f1 is {f1.round(3)}')
        print(f'For selected model AUROC is {yeerokk.round(3)}')
    
    if draw == True:
        fpr, tpr, thresholds = roc_curve(data[3], result)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=yeerokk,
                                  estimator_name='selected_model')
        display.plot()
        plt.show()
    
    return model_upd


# oob_grid ищет комбинации заданных гиперпараметров и, ну, хотя бы пытается оптимизироваться на сабсетах, не попавших в бутстреп-выборки. 

# In[12]:


def oob_grid(data, look_for):

    oob_evolution = []
    best_score = 0
    best_grid = None
    best_rf_bs = None

    well_grown_rf_bs = RandomForestClassifier(bootstrap=True, oob_score=True, random_state=42)

    for choice in ParameterGrid(look_for):
            well_grown_rf_bs.set_params(**choice)
            well_grown_rf_bs.fit(data[0],data[2])
            oob_evolution.append(well_grown_rf_bs.oob_score_)
            if well_grown_rf_bs.oob_score_ > best_score:
                best_score = well_grown_rf_bs.oob_score_
                best_grid = choice
                best_rf_bs = well_grown_rf_bs
    
    print (f'Maximal OOB is {best_score.round(3)}')
    print (f'Grid is {best_grid}')
    
    plt.plot(oob_evolution)
    plt.show()

    return best_rf_bs


# cv_rnd_grid пытается в то же самое, но на рандомизированном поиске, без бутстрэпа, с 4-fold кросс-валидацией по всему скормленному дата-сету.

# In[13]:


def cv_rnd_grid(data,look_for):
    
    second_forest = RandomForestClassifier(bootstrap = False, random_state=42)

    really_random = RandomizedSearchCV(estimator = second_forest, param_distributions = look_for, n_iter = 100, cv = 4, random_state=42, n_jobs = -1)
    really_random.fit(data[0], data[2])

    desired_param_forest = really_random.best_params_
    well_grown_forest = really_random.best_estimator_

    print(f'Best grid is {desired_param_forest}')
    
    return well_grown_forest


# **Запуск чистых моделей.**

# In[14]:


wild_rf_full = RandomForestClassifier(bootstrap = False, random_state=42)
execute(model = wild_rf_full, data = tree, talk = True, draw = True)


# In[15]:


wild_rf_bs = RandomForestClassifier(bootstrap = True, random_state=42)
execute(model = wild_rf_bs,data = tree, talk = True, draw = True)


# **Попытка оптимизации гиперпараметров**

# Если оставаться в рамках разумного, вообще вместо Grid надо было бы подобие Randomized, но и с ним бы полноценный расчёт был бы просто сумасшедшим в плане вычислительного объема, так что большая часть параметров оставлена дефолтной, а полноценно перебираются только число фич, используемых в агрегации, и число деревьев в ансамбле. 

# In[16]:


max_features = ['auto', 0.3, 'log2']
n_estimators = list(range(100,600,10))

look_for = {'max_features': max_features,
            'n_estimators':n_estimators}


# In[17]:


get_ipython().run_cell_magic('time', '', '\nexecute(model = oob_grid(tree, look_for), data = tree, talk = True, draw = True)')


# In[18]:


get_ipython().run_cell_magic('time', '', '\nexecute(model = cv_rnd_grid(tree, look_for), data = tree, talk = True, draw = True)')


# **Подвывод**

# 1. На несбалансированных классах результаты довольно низкие. 
# 2. Попытка файн-тьюна гиперпараметров не увенчалась успехом: в обеих случаях результативность проседает. Варианта два. Либо модель сильно переобучается на валидах, либо так не фортануло, что исходные дефолты были лучше всего пространства предполагаемой "оптимизации". 
# 3. Поскольку число варьируемых параметров существенно увеличивает вычислительную нагрузку, полноценный "прогон" с радикальным улучшением на диване не замутишь. Даже CV пришлось ставить 4-fold. На работе у меня есть доступ к вычислительному кластеру, в следующих проектах я могу попробовать на нем посчитать 5-6 гиперов без мучительного ожидания. Другое дело, что у ревьюеров-то оно будет часов по 6-8 считаться все равно.. 

# ## Борьба с дисбалансом

# **Вновь задаем вначале специфические функции, появляющиеся в этом разделе.**

# upsample/downsample это бессовестные копии с материала курса.

# In[19]:


def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    #выделили где что в фичах
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    #выделили где что в таргетах
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    #объединили после умножения недостаточного класса
    features_upsampled, target_upsampled = shuffle(
    features_upsampled, target_upsampled, random_state=12345)
    #перемешали
    
    return features_upsampled, target_upsampled

def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    #выделили где что в фичах
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    #выделили где что в таргетах
    features_downsampled = pd.concat(
    [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat(
    [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
    #объединили после ужатия избыточного класса
    features_downsampled, target_downsampled = shuffle(
    features_downsampled, target_downsampled, random_state=12345)
    #перемешали
    
    return features_downsampled, target_downsampled


# Создадим соответствующие выборки.

# In[20]:


X_up, y_up = upsample(X, y, 3)

X_red, y_red = downsample(X, y, 0.3)


# In[21]:


train_X_up, test_X_up, train_y_up, test_y_up = train_test_split(X_up,y_up, test_size=0.1, random_state=42)
train_X_red, test_X_red, train_y_red, test_y_red = train_test_split(X_red,y_red, test_size=0.1, random_state=42)

tree_up = (train_X_up, test_X_up, train_y_up, test_y_up)
tree_red = (train_X_red, test_X_red, train_y_red, test_y_red)


# **Запуск чистых моделей**

# In[22]:


wild_rf_full = RandomForestClassifier(bootstrap = False, random_state=42)
execute(model = wild_rf_full, data = tree_up, talk = True, draw = True)
execute(model = wild_rf_full, data = tree_red, talk = True, draw = True)


# In[23]:


wild_rf_bs = RandomForestClassifier(bootstrap = True, random_state=42)
execute(model = wild_rf_bs,data = tree_up, talk = True, draw = True)
execute(model = wild_rf_bs, data = tree_red, talk = True, draw = True)


# **Посмотрим, получится ли оптимизировать на измененных выборках.**

# In[24]:


get_ipython().run_cell_magic('time', '', '\nexecute(model = oob_grid(tree_up, look_for), data = tree_up, talk = True, draw = True)\n\nexecute(model = cv_rnd_grid(tree_up, look_for), data = tree_up, talk = True, draw = True)')


# In[25]:


get_ipython().run_cell_magic('time', '', '\nexecute(model = oob_grid(tree_red, look_for), data = tree_red, talk = True, draw = True)\n\nexecute(model = cv_rnd_grid(tree_red, look_for), data = tree_red, talk = True, draw = True)')


# **Подвывод**

# 1. После балансировки результаты - мое почтение. Приращивание выборки искусственными значениями помогает гораздо больше, чем урезание родной, но, в общем-то, это логично. Больше данных богу данных.
# 2. Опять же, значимо улучшить ситуацию относительно дефолтных значений не удалось, но, впрочем, и ухудшить тоже, так что прогресс, хоть и относительный.
# 3. Значимо улучшается также и AUC-ROC, практически приближаясь к идеальному классифайеру. Если для предыдущих еще имело смысл подбирать какой-то threshold по FPR, исходя из бузнес-логики, то с этими можно прям смело ****як**-****як** - и в продакшн!

# ## Сравнение эффективности методов на сбалансированном микро-сете

# Но. Что, если попробовать провернуть ту же операцию, но на маленьком дата-сете? Какой способ покажет себя лучше?
# 
# Зададим соответствующий дата-сет в 150 значений размером.

# In[32]:


baby_X = X_up.sample(frac=0.011, random_state=42)
baby_y = y_up.sample(frac=0.011, random_state=42)

display(baby_X.info())


# In[27]:


train_X_baby, test_X_baby, train_y_baby, test_y_baby = train_test_split(baby_X,baby_y, test_size=0.1, random_state=42)
baby = (train_X_baby, test_X_baby, train_y_baby, test_y_baby)


# **Чистые модели**

# In[28]:


wild_rf_full = RandomForestClassifier(bootstrap = False, random_state=42)
execute(model = wild_rf_full, data = baby, talk = True, draw = True)


# In[29]:


wild_rf_bs = RandomForestClassifier(bootstrap = True, random_state=42)
execute(model = wild_rf_bs,data = baby, talk = True, draw = True)


# **(((Оптимизация)))**

# In[30]:


get_ipython().run_cell_magic('time', '', '\nexecute(model = oob_grid(baby, look_for), data = baby, talk = True, draw = True)\n\nexecute(model = cv_rnd_grid(baby, look_for), data = baby, talk = True, draw = True)')


# **Подвывод**

# А вот тут уже разница от оптимизации положительная, так как AUC-ROC вырос почти на 1%! То есть классифайер стал хоть капельку, но сильнее. Ну и, опять же, в условиях маленькой выборки бутстрэп значимо outperforming. Хотя общий результат падает достаточно серьезно.  

# ## We need to go smaller

# В случае моей работы дата-сет должен стремиться к минимуму, так как там дорогие физические данные. Попробуем прогнать задачу на совсем маленьком сете, уже только с бутстрэпом.

# In[33]:


nano_X = X_up.sample(frac=0.003, random_state=42)
nano_y = y_up.sample(frac=0.003, random_state=42)

display(nano_X.info())


# In[41]:


wild_rf_bs = RandomForestClassifier(bootstrap = True, oob_score=True, random_state=42)

for i in range(6):
    
    nano_X_test = X_up.sample(frac=0.003, random_state=i)
    nano_y_test = y_up.sample(frac=0.003, random_state=i)
    
    nano = (nano_X, nano_X_test,nano_y,nano_y_test)
    
    execute(model = wild_rf_bs, data = nano, talk = True, draw = False)


# In[38]:


get_ipython().run_cell_magic('time', '', "\nmax_features = ['auto', 0.3, 'log2']\nn_estimators = list(range(100,600,10))\nmax_depth = list(range(2,6))\nmin_samples_leaf = list(range(2,6))\nmin_samples_split = list(range(2,6))\n\nmore_data = {'max_features': max_features,\n            'n_estimators':n_estimators,\n            'max_depth':max_depth,\n            'min_samples_leaf':min_samples_leaf,\n            'min_samples_split':min_samples_split}\n\n#здесь пришлось капельку скостылить\n\nbonsai = oob_grid(data = (nano_X, 'filler',nano_y),look_for = more_data)\n\nfor i in range(6):\n    \n    nano_X_test = X_up.sample(frac=0.003, random_state=i)\n    nano_y_test = y_up.sample(frac=0.003, random_state=i)\n    \n    nano = (nano_X, nano_X_test,nano_y,nano_y_test)\n    \n    execute(model = bonsai, data = nano, talk = True, draw = False)")


# Как итог, мы получаем, что для нано-сета оптимизация по всем гиперпараметрам опять провалилась. Средний f1 для чистого 0.666, а для бoнсая - 0.626, AUROC 0.735 и 0.719, соответственно. Попробуем прогнать по обычному, урезанному набору самых важных: возможно, дерево с ограничениями на гиперы, получается хуже именно потому что интервалы ограничений не слишком обширные, и лучше бы их и вовсе не было. 

# In[42]:


get_ipython().run_cell_magic('time', '', "\nmax_features = ['auto', 0.3, 'log2']\nn_estimators = list(range(100,600,10))\n\nless_data = {'max_features': max_features,\n            'n_estimators':n_estimators}\n\n#здесь пришлось капельку скостылить\n\nbansai = oob_grid(data = (nano_X, 'filler',nano_y),look_for = less_data)\n\nfor i in range(6):\n    \n    nano_X_test = X_up.sample(frac=0.003, random_state=i)\n    nano_y_test = y_up.sample(frac=0.003, random_state=i)\n    \n    nano = (nano_X, nano_X_test,nano_y,nano_y_test)\n    \n    execute(model = bansai, data = nano, talk = True, draw = False)")


# С f1=0.635 и AUROC=0.73 результат улучшился. Возможно, для маленького набора данных, меньшее число деревьев даст лучший результат? 

# In[44]:


get_ipython().run_cell_magic('time', '', "\nmax_features = ['auto', 0.3, 'log2']\nn_estimators = list(range(10,100,2))\n\nless_trees = {'max_features': max_features,\n            'n_estimators':n_estimators}\n\n#здесь пришлось капельку скостылить\n\nbonsai = oob_grid(data = (nano_X, 'filler',nano_y),look_for = less_trees)\n\nfor i in range(6):\n    \n    nano_X_test = X_up.sample(frac=0.003, random_state=i)\n    nano_y_test = y_up.sample(frac=0.003, random_state=i)\n    \n    nano = (nano_X, nano_X_test,nano_y,nano_y_test)\n    \n    execute(model = bonsai, data = nano, talk = True, draw = False)")


# C f1 = 0.664 и AUROC = 0.737 такой прогон уже обошел по второй метрике чистую модель. После я попробовал еще меньше, но такая модель уже начала терять смысл, и немного улучшив f1 сильно уронила AUROC. Также я попробовал такой прогон на дата-сете baby, со 150 значениями, но там выраженного положительного эффекта от снижения числа деревьев не наблюдалось.
# 
# Таким образом, получил занятную гипотезу, что в случае экстремально маленького дата-сета с бутстрепом большое число деревьев начинает буквально повторять друг друга, что приводит к overcorrelation и снижению результативности. В интернете нашел не так много по теме пока, постараюсь еще вникнуть.

# In[39]:


dumbass = DummyClassifier(strategy = 'stratified')
execute(model = dumbass, data = baby, talk = True, draw = True)
execute(model = dumbass, data = nano, talk = True, draw = True)


# Обе модели, для baby и nano, успешно прошли sanity-check, работая лучше случайной модели.

# ## Feature importance

# Собственно, в силу дороговизны данных, как раз и важно понять следующее: как сильно изменение параметра влияет на конкретный дата-сет, чтобы работать, в большей мере, именно с ним.

# In[61]:


oak = RandomForestClassifier(bootstrap = True, random_state=42, max_features = 'auto', n_estimators = 550)
oak_trained = execute(model = oak, data = tree_up, talk = False, draw = False)
display(pd.Series(data = permutation_importance(oak_trained, tree_up[0], tree_up[2], n_repeats=10, random_state=42).importances_mean, index=tree_up[0].columns).sort_values(ascending=False))


# In[62]:


rose = RandomForestClassifier(bootstrap = True, random_state=42, max_features = 'auto', n_estimators = 370)
rose_trained = execute(model = rose, data = baby, talk = False, draw = False)
display(pd.Series(data = permutation_importance(rose_trained, baby[0], baby[2], n_repeats=10, random_state=42).importances_mean, index=baby[0].columns).sort_values(ascending=False))


# In[63]:


bonsai = RandomForestClassifier(bootstrap = True, random_state=42, max_features = 'auto', n_estimators = 48)
bonsai_trained = execute(model = bonsai, data = nano, talk = False, draw = False)
display(pd.Series(data = permutation_importance(bonsai_trained, nano[0], nano[2], n_repeats=10, random_state=42).importances_mean, index=nano[0].columns).sort_values(ascending=False))


# Важным выводом является то, что ключевой признак определяется даже при дата-сете в 240 раз меньше оригинального. Но вот с общей точностью, конечно, в таком случае проблемы, ведь просто недостаточно примеров может попасть в обозреваемый сет. 
# 
# Поскольку везде возраст значим, давайте еще и какую-то бизнес-рекомендацию выдадим: посмотрим на средний возраст и счёт уходящих и остающихся.

# In[66]:


stay_age = based.loc[based['Exited']==0,'Age'].mean()
leave_age = based.loc[based['Exited']==1,'Age'].mean()

stay_m = based.loc[based['Exited']==0,'Balance'].mean()
leave_m = based.loc[based['Exited']==1,'Balance'].mean()

print(f'Stay is {stay_age.round(1)} yrs old, leave is {leave_age.round(1)} yrs old')
print(f'Stay is {stay_m.round(2)} euro, leave is {leave_m.round(2)} euro')


# Похоже, что у Бета-Банка серьезные проблемы с немецкими пенсионерами, выводящими свои рейхсмарки! Солидная публика, за такую имеет смысл побороться.

# ## Общий вывод

# 1. Терпение и труд - все я устал. 
# 2. Все поставленные в ТЗ задачи достигнуты.
# 3. Никак не получается у меня освоить хоть какой-то способ оптимизации гиперпараметров: как не крути, модели после вроде бы файн-тьюна лучше не получаются. 
# 4. Впервые в жизни столкнулся с тем, что компухтер не моментально исполняет твои барские пожелания.
# 5. Изучил, как работают деревья в случае экстремально-маленьких дата-сетов, что важно для моей работы, где сразу побегу знания внедрять, но все еще остается без ответа ряд гипотез. 
# 
# Буду рад фидбэку и споесба вам за внимание <3

# ## Чек-лист готовности проекта

# Поставьте 'x' в выполненных пунктах. Далее нажмите Shift+Enter.

# - [x]  Jupyter Notebook открыт
# - [x]  Весь код выполняется без ошибок
# - [x]  Ячейки с кодом расположены в порядке исполнения
# - [x]  Выполнен шаг 1: данные подготовлены
# - [x]  Выполнен шаг 2: задача исследована
#     - [x]  Исследован баланс классов
#     - [x]  Изучены модели без учёта дисбаланса
#     - [x]  Написаны выводы по результатам исследования
# - [x]  Выполнен шаг 3: учтён дисбаланс
#     - [x]  Применено несколько способов борьбы с дисбалансом
#     - [x]  Написаны выводы по результатам исследования
# - [x]  Выполнен шаг 4: проведено тестирование
# - [x]  Удалось достичь *F1*-меры не менее 0.59
# - [x]  Исследована метрика *AUC-ROC*
