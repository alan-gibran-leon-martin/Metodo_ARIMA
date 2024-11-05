import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima

#Limpieza de datos
yen= pd.read_csv("C:/Users/zaira/Desktop/alan/series_temporales/dolar/yen.csv", header= None)
yen= yen.drop([1])
yen[0]= pd.to_datetime(yen[0], format="%d/%m/%Y", errors='coerce')
yen[1]= pd.to_numeric(yen[1], errors="coerce")
yen= yen.dropna()
yen2= yen
#Prueba Dickey-Fuller (estacionalidad)
pdf= adfuller(yen[1])
print("Prueba Dickey-Fuller")
print(f"Estadistica ADF: {pdf[0]}") #Esto se compara con los valores de significacia, si es menor entonces es estacionaria
print(f"Valor p:{pdf[1]} ")
print(f"No. de rezagos utilizados: {pdf[2]}")
print(f"No. de observaciones; {pdf[3]}")
print("Valores crìticos")
for key, value in pdf[4].items():
    print(f"{key}, {value}")
#No hay estacionalidad
#Primera diferenciaciòn
d1yen= yen[1].diff().dropna()
#Prueba Dickey-Fuller 
pdf2= adfuller(d1yen)
print("\nSegunda prueba Dickey-Fuller")
print(f"Estadistica Dickey-Fuller: {pdf2[0]}")
print(f"Valor p: {pdf2[1]}")
print(f"No de rezagos utilizados: {pdf2[2]}")
print(f"No. de observaciones: {pdf2[3]}")
print("Valores crìticos")
for key, value in pdf2[4].items():
    print(f"{key}, {value}")
# Ya que el valor de la estadistica Dickey-Fuller: -13.29 y el valor de 
# significancia en 5% es stadistica Dickey-Fuller: -13.29 hay estacionalidad despues de la primera diferenciacion
#Graficar primera diferenciacion
#Grafico original
fig, ax = plt.subplots(3,2, sharex= False, sharey= False)
fig.suptitle("Grafica de estacionalidad")
ax[0,0].plot(yen[0], yen [1])
plot_acf(yen[1], ax=ax[0,1])
#Graficar primera diferenciacion
ax[1,0].plot(yen[0][1:],d1yen)
plot_acf(d1yen, ax=ax[1,1])
plot_pacf(d1yen, ax= ax[2,0])
plt.tight_layout()#Sirve para ajustar los subgraficos
#Solo es necesaria una diferenciacion d= 1, autocorrelaciòn sobresale un solo dato q=1, 
# y solo un dato sobresale en autocorrelaciòn parcial p=1

#Modelo ARIMA
arima1= ARIMA(yen[1].values, order=(1,1,1))
arima1= arima1.fit()
print(arima1.summary())
#Obtener residuales
residuos= pd.DataFrame(arima1.resid)
#Prueba Ljung box (ruido blanco)
pljung= acorr_ljungbox(residuos, lags=[10], return_df=True)
print(pljung) # ya que el valor de p (lb_pvalue) > que 0.05 se rechaza h0 por lo 
#que no hay autocorrelacion (es informaciòn no valiosa para el modelo= ruido blanco)
fig, ax2 = plt.subplots(1,2, sharex= False, sharey= False)
residuos.plot(title="Residuales", ax= ax2[0])
residuos.plot(kind="kde", title= "Densidad", ax=ax2[1] )
#Graficar la linea original junto con la predicciòn del modelo ARIMA
prediccion= arima1.predict(dynamic= False)
ax[2,1].plot(yen[0], yen[1], label= "Datos originales", color= "Blue")
ax[2,1].plot(yen[0],prediccion,linestyle="--", label= "Predicciòn", color= "Red")
ax[2,1].set_title("ARIMA")
ax[2,1].legend(loc= "best")

#Dividir en train y test
yen=yen.set_index(0)
x= yen[1]
train, test= x[:-200], x[-200:]
itrain, itest= yen.index[0:-200], yen.index[-200:]

#Prueba del modelo y correr el modelo con los datos de entrenamiento
arima2= ARIMA(train.values, order=(1,1,1))
arima2= arima2.fit()
#obtener la predicciòn y el intervalo de confianza
fc= arima2.get_forecast(steps=200)
intcon= fc.conf_int(alpha=0.05)
#Ponerle el mismo ìndice a la predicciòn y los intervalos de confianza
fc_series= pd.Series(fc.predicted_mean, index=itest)
lower_series= pd.Series(intcon[:,0], index= itest)
upper_series= pd.Series(intcon[:,1], index=itest)
#print(arima2.summary())

#Modelo auto-ARIMA
autoarima= auto_arima(train, start_p= 0, start_q= 0, max_p= 5, max_q=5, seasonal= False, trace= True, stepwise= True, suppress_warnings= True)
print(autoarima.summary())
#Obtener predicciòn y el intervalo de confianza
fc2, intconf2= autoarima.predict(n_periods= 200, return_conf_int= True)
fc2_serie= pd.Series(fc2)
#Valores superior e inferior de intervalo de confianza
lower_series2= pd.Series(intconf2[:,0], index=itest)
upper_series2= pd.Series(intconf2[:,1], index= itest)

#Graficar los datos de entrenamiento, la predicciòn y los intervalos de confianza del modelo ARIMA(1,1,1)
fig, ax3 = plt.subplots(2,1)
fig.suptitle("Comparaciòn modelo ARIMA y autoARIMA")
ax3[0].plot(itrain, train, label= "Entrenamiento")
ax3[0].plot(itest, test, label= "Actual")
ax3[0].plot(fc_series, label= "Prediccion")
ax3[0].fill_between(lower_series.index, lower_series, upper_series, color= "Blue", alpha= 0.05)
ax3[0].legend(loc="best")
ax3[0].set_title("Prueba del modelo ARIMA")

#Graficar modelo autoARIMA
ax3[1].plot(itrain, train, label= "Entrenamiento")
ax3[1].plot(itest, test, label= "Actual")
ax3[1].plot(itest, fc2_serie, label= "Predicciòn")
ax3[1].fill_between(lower_series2.index, lower_series2, upper_series2, color="Blue", alpha=0.05)
ax3[1].legend(loc= "best")
ax3[1].set_title("Prueba del modelo autoARIMA")
plt.tight_layout()

#Generar predicciòn
autoarima2= auto_arima(yen, start_p= 0, start_q=0, max_p=5, max_q=5, seasonal= False,trace=True, stepwise= True, suppress_warnings=True)
#print(autoarima2.summary())
daterange= pd.Series(pd.date_range(start=yen2.iloc[-1,0], periods= 200, freq="D"))
fc3, intconf3 = autoarima2.predict(n_periods= 200, return_conf_int= True)
fc3_serie= pd.Series(fc3)
lower_series3= pd.Series(intconf3[:,0], index= daterange)
upper_series3= pd.Series(intconf3[:,1], index= daterange)
#Graficar predicciòn
fig, ax4= plt.subplots()
ax4.plot(yen2[0],yen2[1], label= "Datos originales")
ax4.plot(daterange, fc3, label= "Predicciòn", color= "Red")
ax4.fill_between(lower_series3.index, lower_series3, upper_series3, color= "Blue", alpha=0.05)
ax4.legend(loc="best")
ax4.set_title("Predicciòn con los datos completos")
plt.show()