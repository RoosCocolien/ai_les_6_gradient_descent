"""
GRADIENT DESCENT
We gebruiken het lineaire regressiemodel ŷ = a + b * x
met de gradient descent methode om de coefficienten
a en b te trainen.
DOEL: vindt de lijn (ŷ = a + b * x) dat het beste overeenkomt met de data,
zodat we kunnen kijken of er een afhankelijkheid is
ŷ is de voorspelde waarde van de afhankelijke variabele (in ons voorbeeld happiness)
a is de intercept (de waarde van de afhankelijke variabele y wanneer de onafhankelijke
variabele x is zeo
b is de slope
x is de onafhankelijke variabele (income)

ONTVANG X: lijst met onafhankelijke waarden van de observaties
(denk aan aanwezigheid/inkomen)
ONTVANG Y: lijst met afhankelijke waarden van de observaties
(denk aan cijfer/geluk)
ONTVANG NUM_ITERATIONS: int, aantal iteraties om te leren (vb: 10.000)
ONTVANG LEARNING_RATE: float, leerconstante (vb: 0.0001)
INSTANTIALISEER COEFFICIENTS A EN B op vb 0

HERHAAL VOOR NUM_ITERATIES
    HERHAAL VOOR ELKE OBSERVATIE (Xk,Yk)
        BEREKEN VOORSPELLING = A + B * Xk
        BEREKEN ERROR = VOORSPELLING - Yk
        COEFFICIENT A AANPASSEN (INTERCEPT) --> A - ERROR * LEARNING_RATE
        COEFFICIENT B AANPASSEN (SLOPE) --> B - Xk * ERROR * LEARNING_RATE
RETURN COEFFICIENTS

Data: https://www.scribbr.com/statistics/simple-linear-regression/

"""

# TODO: je moet zelf een gradient_descent functie schrijven en importeren
from ai_6_lineaire_regressie import gradient_descent
# met pyplot kunnen we een eenvoudige lijnplot maken
import matplotlib.pyplot as pyplot
# we gaan pandas gebruiken, dat is een open-source python lib
# wordt gebruikt voor data-analyse en data-manipulatie
# met pandas kunnen we data inlezen (oa csv), manipuleren en analyseren
# we kunnen een csv bestand inlezen en omzetten naar een tabelachtige structuur (dataFrame)
# we kunnen kolommen ook hernoemen
import pandas
# om uit een dataframe structuur een column te extraheren als array gebruiken we
# arrays maken en manipuleren / wiskundige bewerkingen / wiskundige functie / genereren van data etc.
import numpy

# inlezen csv data met pandas
dataFrameTable = pandas.read_csv("data/income.data.csv")
pyplot.scatter(dataFrameTable.income, dataFrameTable.happiness)
print(dataFrameTable['income'])
incomeArray = numpy.array(dataFrameTable.income)
# print(incomeArray)
happinessArray = numpy.array(dataFrameTable.happiness)
# print(happinessArray)

# scatter plot maken
pyplot.scatter(incomeArray, happinessArray)
pyplot.savefig("media/data_income_happiness_example.png")

# gradient descent
# inkomen is de onafhankelijke waarde
# happiness is de afhankelijke waarde
a, b = gradient_descent(incomeArray, happinessArray)
print(f"mijn a, b: {a} en {b}")
# maak de lineaire vergelijking y = ax + b
# a is de helling (m), b is de intercept (b)
vergelijking = f"y = {a:.2f} + {b:.2f} * x"
print(vergelijking)
# teken de lijn op een grafiek
# dit is de titel
pyplot.title(f"Title: {vergelijking}")
pyplot.xlabel("income x 1000")
pyplot.ylabel("happiness")
# sla de minimale en maximale waarden van de dataframe op
# x_min, x_max = min(dataFrameTable.income), max(dataFrameTable.income)
x_min, x_max = dataFrameTable['income'].min(), dataFrameTable['income'].max()
print(x_min, x_max)
# de y waarden moeten berekend worden met de lineaire vergelijking y=a + b * x
# hiermee worden de coordinaten van de lijn op de grafiek
y_min = a + b * x_min
y_max = a + b * x_max
# hierbij is a de intercept (waar y met 0 snijdt, en b de helling)

pyplot.plot([x_min, x_max], [y_min, y_max], c='red')
pyplot.savefig("media/data_income_happiness_lin_red_example.png")
pyplot.clf()




