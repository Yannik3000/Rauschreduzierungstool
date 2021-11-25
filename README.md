# Rauschreduzierungstool
 
 Der Aufruf des Tools muss mit einer IP-Adresse von einem Timeswipe als Argument erfolgen. Zum Beispiel: python Rauscheinstellungen_GUI.py ws://10.0.0.107:8888
 
 ### Signal Einlesen: 
 
 Mit klick auf "Record reference signal" kann dann ein Referenzsignal eingelesen werden, anhand dessen ein Rauschmodell bestimmt wird. Wurde ein Referenzsignal eingelesen, kann danach ein Nutzsignal eingelesen werden, um die Rauschreduktion darauf zu überprüfen. Vor dem Einlesen der Daten kann die Länge des Signals in Sekunden und der Kanal eingestellt werden.
 
 ### Rauscheinstellungen:
 
 Es Können folgende zwei Parameter für die Rauschreduktion eingestellt werden. Um das Signal erneut mit diesen Einstellungen zu komprimieren, muss der Button "Update Plot" betätigt werden.
 
##### Sigma Wert:
 
 Der Sigma Wert gibt die Stärke der Rauschreduzierung an. Übliche Werte sind im Bereich von 0 bis 8. Kommazahlen müssen mit Punkt geschrieben werden.
 
##### Slice Size:
 
 Bei der Rauschreduktion wird das Signal in Slices verarbeitet. Mit der "Slice Size" Einstellung kann die Größe dieser Slices angepasst werden.
 
### Plots:
 
 Die oberen beiden Plots zeigen das Referenzsignal, die unteren das Nutzsignal. Auf der linken Seite befindet sich jeweils das Original Signal auf der rechten Seite das komprimierte Signal. Mit "Size of plottet window" kann die Größe des dargestellten Teils des Signals angepasst werden.
 
### Regressionskurve:
 
 Mit dem Button "Show Regression Curve" kann das Rauschmodell betrachtet werden.
 
### Speichern:
 
 Mit dem Button "Save" können die Parameter des Rauschmodells gespeichert werden. Dafür muss ein Ordner mit dem Namen "Calibration" im gleichen Ordner, indem das Skript liegt, angelegt werden. Dort wird dann eine JSON-Datei mit den Parametern des Rauschmodells angelegt.
