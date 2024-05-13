# ProjectionMappingBP
## Logboek
| Datum      | Beschrijving |
|------------|-----------------------------------------------------------------------------------------------------------------------|
| 21-02-2024 | Meeting + start met schrijven van camera calibratie code |
| 22-02-2024 | Finished calibratie code + start tracking van markers en vlakken code schrijven |
| 23-02-2024 | Finished tracking code van markers en vlakken, started schrijven van 3D tracking van doos met markers met overdracht van data naar unity, gedeeltelijk werkende |
| 25-02-2024 | Verder gewerkt aan tracking en prediction code |
| 28-02-2024 | Verder gewerkt aan tracking en prediction code |
| 29-02-2024 | Verder gewerkt aan tracking en prediction code |
| 01-03-2024 | Prediction en 3D wereldcoördinaten code afgewerkt! |
| 07-03-2024 | Schrijven van code om getrackte object van opencv naar unity te vertalen, werkende zonder prediction |
| 11-03-2024 | Toevoegen prediction aan vertaling van opencv naar unity gestart |
| 13-03-2024 | Verder gewerkt aan toevoegen van prediction aan vertaling |
| 14-03-2024 | Vertalen met prediction afgewerkt en werkende, geeft soms inconsistente resultaten -> (morgen) vragen wat ik hier aan kan doen? |
| 20-03-2024 | Verder gekeken naar kalmanfilter, werkt nog niet. Joni ging er ook eens naar kijken, of vragen aan Nick wanneer mogelijk. Begonnen aan camera projector kalibratie, nog geen code, eerst researchen. |
| 21-03-2024 | Verder research gedaan naar kalibratie, 2 methodes gevonden vergelijken en beslissen welke implementeren. |
| 26-03-2024 | Meeting + beslist om ray plane intersect te gebruiken omwille van mogelijke onnauwkeurigheid (als ik het zelf doe) van de graycode methode uit afgelopen les en taak 2. |
| 27-03-2024 | Verder research gedaan naar hoe kalibratie code werkt, begonnen aan de implementatie van ray plane intersect. |
| 28-03-2024 | Verder gewerkt aan ray plane intersect implementatie. |
| 17-04-2024 | Verder gewerkt aan ray plane intersect implementatie. Het werkt niet ;( |
| 18-04-2024 | Verder gewerkt aan ray plane intersect implementatie. Het werkt niet ;( |
| 19-04-2024 | Verder gewerkt aan ray plane intersect implementatie. Het werkt niet ;( |
| 22-04-2024 | Verder gewerkt aan ray plane intersect implementatie. Extrensieke parameters werken bijna... |
| 23-04-2024 | Verder gewerkt aan ray plane intersect implementatie. Extrensieke parameters zijn correct, maar detectie is nog instabiel. Gecontroleerd met visualisatie die ik geschreven heb. |
| 24-04-2024 | Verder gewerkt aan ray plane intersect implementatie. Extrensieke parameters zijn correct en stabiel! Gecontroleerd met visualisatie. |
| 25-04-2024 | Verder gewerkt aan ray plane intersect implementatie. Geprobeerd om dynamische kalibratie te gebruiken om intrinsieke parameters van projector te bepalen, nog niet gelukt. |
| 26-04-2024 | Verder gewerkt aan ray plane intersect implementatie. Veranderd naar statische kalibratie voor de intrinsieke parameters te bepalen aangezien camera en projector ook blijven staan op hun plek. |
| 01-05-2024 | Verder gewerkt aan ray plane intersect implementatie. Methode van afgelopen week herdacht want implementatie werkt niet zoals verwacht. |
| 02-05-2024 | Verder gewerkt aan nieuwe implementatie, twijfels als deze wel correct zal zijn. |
| 03-05-2024 | Nogmaals opnieuw gekeken naar een juist implementatie, de vorige werkte ook niet als verwacht. |
| 04-05-2024 | Verder gewerkt aan nieuwe implementatie. |
| 11-05-2024 | Online gesproken met Joni over wat er precies fout gaat en als mijn denkwijze correct is. Hij heeft een voorstel van een implementatie gedaan, implementatie al geprobeerd. 13-05-2024 naar het edm om de methode nog eens duidelijk te overlopen. |
| 12-05-2024 | Al zo ver mogelijk geprobeerd de implementatie correct te doen. |
| 13-05-2024 | Uitleg gekregen van Joni, het idee en de methode achter de 'juiste' implementatie is nu duidelijk voor me. De implementatie geschreven en in theorie werkt het. In praktijk zijn de resultaten nog niet heel correct. Kan opgelost worden door patroon in verschillende plaats van het beeld te zetten, bijvoorbeeld ook eens in de hoeken. |
