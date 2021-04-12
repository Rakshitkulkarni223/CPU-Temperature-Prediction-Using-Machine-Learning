import csv
import wmi

colums=["CPU Total","CPU Package Temperature","RAM","Disk Temperature","Used Space","Power CPU Package","Power CPU Cores","Power CPU Graphics","Power CPU DRAM"]

dataset=open("data/test_data.csv","w",newline='')
writer = csv.writer(dataset)
writer.writerow(colums)

w = wmi.WMI(namespace="root\OpenHardwareMonitor")
infos = w.Sensor()

for i in range(5000):

    details=[-1]*9

    for sensor in infos:

        if sensor.SensorType=="Load":
            if (sensor.Name=="CPU Total"):
                details[0]=sensor.Value

        if sensor.SensorType == 'Temperature':
            if sensor.Name=="CPU Package":
                details[1]=sensor.Value

        if sensor.SensorType == 'Load':
            if sensor.Name=="Memory":
                details[2]=sensor.Value

        if sensor.SensorType=="Temperature":
            if sensor.Name=="Temperature" and sensor.Parent=="/hdd/0":
                details[3]=sensor.Value

        if sensor.SensorType=="Load":
            if sensor.Name=="Used Space" and sensor.Parent=="/hdd/1":
                details[4]=sensor.Value

        if sensor.SensorType == 'Power':
            if sensor.Name == "CPU Package":
                details[5] = sensor.Value

        if sensor.SensorType == 'Power':
            if sensor.Name == "CPU Cores":
                details[6] = sensor.Value

        if sensor.SensorType == 'Power':
            if sensor.Name == "CPU Graphics":
                details[7] = sensor.Value

        if sensor.SensorType == 'Power':
            if sensor.Name == "CPU DRAM":
                details[8] = sensor.Value

    writer.writerow(details)
    w = wmi.WMI(namespace="root\OpenHardwareMonitor")
    infos = w.Sensor()

dataset.close()