import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
import pmdarima as pm
from numpy import log
from pmdarima import auto_arima

#Selecciòn y limpieza de datos
dolar= pd.read_csv("C:/Users/zaira/Desktop/alan/series_temporales/dolar/dolaar.csv", header=None)
dolar[1]= pd.to_numeric(dolar[1], errors="coerce")
dolar= dolar.drop([0])
dolar[0]= pd.to_datetime(dolar[0], dayfirst= True)
dolar = dolar.dropna()

#Prueba de estacionalidad
prueba= adfuller(dolar[1].values)
print("\n\nPrueba Dickey-Fueller\n")
print(f"Estadistica ADF: {prueba[0]}")
print(f"Valor p: {prueba[1]}")
print(f"No. de rezagos utilizados: {prueba[2]}")
print(f"No. de observaciones {prueba[3]}")
print("Valores criticos")
for key, value in prueba[4].items():
    print(f" {key}: {value}")


#Creaciòn del gràfico original
fig, ax = plt.subplots(3,2, sharex= False, sharey= False)
fig.suptitle("Revisiòn de estacionalidad")
ax[0,0].plot(dolar[0], dolar[1], label= "Dolar"); ax[0,0].set_title("Valor del dolar en el tiempo")
plot_acf(dolar[1], ax= ax[0,1])

#Primera diferenciaciòn
edolar= dolar.diff().dropna()
ax[1,0].plot(dolar[0][1:], edolar[1]); ax[1,0].set_title("Primera diferenciaciòn")
plot_acf(edolar[1], ax= ax[1,1])

#Segunda diferenciaciòn
edolar2= dolar.diff().diff().dropna()
ax[2,0].plot(dolar[0][2:], edolar2[1]); ax[2,0].set_title("Segunda diferenciaciòn")
plot_acf(edolar2[1], ax= ax[2,1])

plt.tight_layout()#Sirve para ajustar los subgraficos

#Prueba de estacionalidad en la primera diferenciaciòn
print("\n\nValores de la prueba de la primera diferenciaciòn\n")
prueba2= adfuller(edolar[1].values)
print(f"Estadistica ADF: {prueba2[0]}")
print(f"Valor p; {prueba2[1]}")
print(f"No. rezagos utilizados: {prueba2[2]}")
print(f"No. de observaciones : {prueba2[3]}")
print("Valores criticos")
for key, value in prueba2[4].items():
    print(f"{key}:{value}")

#Prueba de estacionalidad en la segunda diferenciaciòn
print("\n\nValores de la prueba de la segunda diferenciaciòn\n")
prueba3= adfuller(edolar2[1].values)
print(f"Estadistica ADF: {prueba3[0]}")
print(f"Valor p; {prueba3[1]}")
print(f"No. rezagos utilizados: {prueba3[2]}")
print(f"No. de observaciones : {prueba3[3]}")
print("Valores criticos")
for key, value in prueba3[4].items():
    print(f"{key}:{value}")

#El valor de "d" es el numero de veces que es necesario aplicar la diferenciacion para lograr estacionalidad
#El valor "d" en este caso es 1.

#Autocorrelaciòn parcial PACF
fig, ax = plt.subplots(3,2, sharex= False, sharey= False)
fig.suptitle("Ajuste del ARIMA")
plot_pacf(dolar[1].diff().dropna(), ax=ax[0,0])
ax[0,0].set_xlabel("Rezago")
ax[0,0].set_ylabel("Autocorrelación parcial") 
plot_acf(dolar[1].diff().dropna(), ax=ax[0,1])
ax[0,1].set_xlabel("Rezago")
ax[0,1].set_ylabel("Autocorrelación") 
#El valor PACF(p) es igual a 1 porque solo hay una rezago que se sale de las lineas de significancia
#Media movil MA(q) obtiene un valor de 1 puesto que en acf no hay rezagos sobresalientes 

#Modelo ARIMA
arima1 = ARIMA(dolar[1].values, order =(1,1,1))
arima1 = arima1.fit()
print(arima1.summary())

#Prueba de Ljung Box
ljung_box_results = acorr_ljungbox(arima1.resid, lags=[10], return_df=True)
print(ljung_box_results)
#El resultado es 1.079952e-11 por lo que se acepta la hipotesis nula  y se afirma que hay ruido blanco

#Obtener residuales
residuo= pd.DataFrame(arima1.resid)
residuo.plot(title="Residuales", ax= ax[1,0], lw=2)
residuo.plot(kind= "kde", title= "Densidad", ax= ax[1,1], lw=2)

#Grafica de laprediccion

predictions = arima1.predict(dynamic=False)

# Graficar los datos reales y las predicciones

ax[2,0].plot(dolar[1], label='Datos Reales', color='red')  # Los datos originales
ax[2,0].plot(predictions, label='Predicciones ARIMA', color='blue', linestyle= '--')  # Las predicciones
ax[2,0].set_title('Predicción con ARIMA')
ax[2,0].legend(loc='best')

## Prueba del modelo
X = dolar[1]
train, test = X[0:-200], X[-200:]

## Ajustar el modelo ARIMA
arima2 = ARIMA(train, order=(1,1,1))
arima2 = arima2.fit()

## Realizar la predicción
fc = arima2.get_forecast(steps=200)
conf = fc.conf_int(alpha=0.05)  # Intervalo de confianza

## Convertir las predicciones a series con el mismo índice que test
fc_series = pd.Series(fc.predicted_mean, index=test.index)
lower_series = pd.Series(conf.iloc[:, 0], index=test.index)
upper_series = pd.Series(conf.iloc[:, 1], index=test.index)

## Graficar la prueba del modelo
ax[2,1].plot(train, label='Training', lw=2)
ax[2,1].plot(test, label='Actual', lw=2)
ax[2,1].plot(fc_series, label='Forecast', lw=2)
ax[2,1].fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=0.15)#fill es para rellenar y se usa para las bandas de confianza
ax[2,1].legend(loc='upper left')
ax[2,1].set_title('Prueba del modelo ARIMA (1,1,1)')
plt.tight_layout()

modelo_auto = auto_arima(train, start_p=0, start_q=0, max_p=5, max_q=5, seasonal=False, trace=True, stepwise=True, suppress_warnings=True)
print(modelo_auto.summary())


# Realizar la predicción 
n_periods = 200
fc2 = modelo_auto.predict(n_periods= n_periods)# Número de pasos a predecir

# Crear un DataFrame para los intervalos de confianza 
conf2 = modelo_auto.predict(n_periods=n_periods, return_conf_int=True)
fc_series2 = pd.Series(fc2, index=test.index[:n_periods])

# Extraer los límites superior e inferior del intervalo de confianza
lower_series2 = pd.Series(conf2[1][:, 0], index=test.index[:n_periods])
upper_series2 = pd.Series(conf2[1][:, 1], index=test.index[:n_periods])

# Gràficar la prueba del modelo ARIMA
fig, ax = plt.subplots(1,2, sharex= False, sharey=False)
fig.suptitle("Comparaciòn entre ARIMA y auto-ARIMA")
ax[0].plot(train, label='Training', lw=2)
ax[0].plot(test, label='Actual', lw=2)
ax[0].plot(fc_series, label='Forecast', lw=2)
ax[0].fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=0.05)
ax[0].legend(loc='upper left')
ax[0].set_title('Prueba del modelo ARIMA (1,1,1)')

#Gràfica del modelo auto-ARIMA
ax[1].plot(train, label='Training', lw=2)
ax[1].plot(test, label='Actual', lw=2)
ax[1].plot(fc_series2, label='Forecast', lw=2)
ax[1].fill_between(lower_series2.index, lower_series2, upper_series2, color='k', alpha=0.05)
ax[1].legend(loc='upper left')
ax[1].set_title('Prueba del modelo auto-ARIMA')
plt.tight_layout()

#Modelo auto-ARIMA con los datos completos
modelo_auto2 = auto_arima(dolar[1], start_p=0, start_q=0, max_p=5, max_q=5, seasonal=False, trace=True, stepwise=True, suppress_warnings=True)
#print(modelo_auto2.summary())

# Realizar la predicción 
n3_periods= 200
fc3, conf3 = modelo_auto2.predict(n_periods= n3_periods, return_conf_int=True)# Número de pasos a predecir

# Crear un DataFrame para los intervalos de confianza 

fc3= pd.Series(fc3).reset_index(drop=True)
date_range= pd.Series(pd.date_range(start=dolar.iloc[-1,0] +pd.Timedelta(days= 1), periods=n3_periods, freq='D'))
fc_series3= pd.concat([date_range,fc3], axis=1)
# Extraer los límites superior e inferior del intervalo de confianza
lower_series3 = pd.Series(conf3[:, 0], index=fc_series3[0])
upper_series3 = pd.Series(conf3[:, 1], index=fc_series3[0])
#Gràfica del modelo auto-ARIMA
fig, ax4= plt.subplots(1)
ax4.plot(dolar[0], dolar[1], label='Precios reales', lw=2)
ax4.plot(fc_series3[0], fc_series3[1], label='Predicciòn', lw=2, color='Red')
ax4.fill_between(lower_series3.index, lower_series3.values, upper_series3.values, color='Blue', alpha=0.05)
ax4.legend(loc='upper left')
ax4.set_title('Prueba del modelo auto-ARIMA con los datos completos')

plt.show()

