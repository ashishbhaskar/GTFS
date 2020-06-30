from google.transit import gtfs_realtime_pb2
import urllib
import urllib.request
import time
import datetime
import os
import multiprocessing
import re
import pandas as pd

Working_Directory = r"//rstore.qut.edu.au/projects/sef/hetrogenmodel/Data/GTFS/"   #ensure slash sign in the end "/" not "\"

def monthname(mydate):
    mydate = datetime.datetime.now()
    m = mydate.strftime("%B")
    return(m)

def Brisbane(epoch):
    a = time.strftime("%d-%m-%Y %H:%M:%S", time.localtime(epoch))
    return(a)

def createCSVfile(inputlist , name):
    with open( name , 'w') as f:
        for i in inputlist:
            k = 0
            for item in i:  
                f.write(str(item) + ',')
                k = k+1
            f.write('\n')
        return(f)
def inputCSVfile(csvfile):
    list1= []
    with open(csvfile, 'r') as f:
        for i in f:
            j = i.split(',')
            
            le = len(j)
            j[le - 1] = (j[le- 1]).strip()
            list1.append(j)
        return(list1)

def createCSVfileWCD(inputlist , name):
    with open( name , 'w') as f:
        for i in inputlist:
            f.write(str(i))
            f.write('\n')
        return(f)
def SpeedProcessing(A1, Working_DR, BB1, CC1, names):
    P = []
    for indexo, ki in enumerate(A1):
        A1['Trip Id Unique'] = A1['Trip ID'].astype(str) + A1['Star t Time'].astype(str) + A1['Start Date'].astype(str)+ A1['Route_ID'].astype(str)
        A1['HistPred'] = pd.to_numeric(A1['timestamp']) - pd.to_numeric(A1['Arrival Time (UNIX)'])
        A1 = A1[(A1['HistPred'] > -600)]
        A1['Route_ID'] = A1['Route_ID'].str.extract('(.*)-\d+').astype(str)
        BB1['Route ID'] = BB1['Route ID'].astype(str)
        BB1['Stop ID'] = BB1['Stop ID'].astype(str)
        BB1['Stop Seq'] = BB1['Stop Seq'].astype(str)
        D1 = pd.merge(A1,BB1, how = 'inner', left_on = ['Route_ID', 'Stop Sequence', 'stop_id'], right_on = ['Route ID', 'Stop Seq', 'Stop ID'])
        D1['Shape ID'] = D1['Shape ID'].astype(str)
        D1['Stop Sequence'] = D1['Stop Sequence'].astype(str)
        CC1['Shape_ID '] = CC1['Shape_ID '].astype(str)
        CC1['Stop Sequence (Origin)'] = CC1['Stop Sequence (Origin)'].astype(str)
        E1 = pd.merge(D1,CC1, how = 'inner', left_on = ['Shape ID', 'Stop Sequence'], right_on = ['Shape_ID ', 'Stop Sequence (Origin)'])
        F1 = E1.copy(deep=True)
        E1['Stop Sequence (Destination)'] = E1['Stop Sequence (Destination)'].astype(str)
        CC1['Stop Sequence (Origin)'] = CC1['Stop Sequence (Origin)'].astype(str)
        G1 = pd.merge(E1,F1, how = 'inner', left_on = ['Trip Id Unique', 'Stop Sequence (Destination)'], right_on = ['Trip Id Unique', 'Stop Sequence (Origin)'])
        H1 = G1[['Trip Id Unique', 'timestamp_x', 'Shape_ID _x', 'Distance beween Stops (km)_x', 'Arrival Time (UNIX)_x', 'Departure Time (UNIX)_x', 'Arrival Time (UNIX)_y', 'Departure Time (UNIX)_y']].copy()
        H1['A2A_speed'] = (H1['Distance beween Stops (km)_x'].astype(float)*60*60)/(H1['Arrival Time (UNIX)_y'].astype(float) - H1['Arrival Time (UNIX)_x'].astype(float))
        H1['D2D_speed'] = (H1['Distance beween Stops (km)_x'].astype(float)*60*60)/(H1['Departure Time (UNIX)_y'].astype(float) - H1['Departure Time (UNIX)_x'].astype(float))
        H1['clear_speed'] = ((H1['Distance beween Stops (km)_x'].astype(float)-.12)*60*60)/(H1['Arrival Time (UNIX)_y'].astype(float) - H1['Departure Time (UNIX)_x'].astype(float))
        H1['SS Id'] = G1['Stop ID_x'].map(str) + '_' + G1['Stop ID_y'].map(str)
        H1['Delay'] = G1['Arrival Delay_y'].map(float) - G1['Arrival Delay_x'].map(float)
        H1['Stop Seq'] = G1['Stop Sequence (Origin)_x'].map(str) + '_' +  G1['Stop Sequence (Destination)_x'].map(str)
        H1['Route_Id'] = G1['Route_ID_x']
        if indexo ==0:
            P = H1.copy(deep = True)
        if indexo>0:
            P = P.append(H1)
        if indexo%20 == 0:
            print(indexo)
    P = P.drop_duplicates(keep=False)
    P.to_csv(Working_DR + '/'+ 'Speed ' +names + '.csv', index=False, header=True)    
def Processing1(A, directory2, directory3, a, fiction):
    t = float(time.time())
    createCSVfileWCD(A, directory2 + "/"  +a+str(t)+'_'+'.csv') #'Trip_Update'

#(Trip_Update[n], TUdirectory2, TUdirectory20, hhhh, fiction, BB, CC, TUdirectory30,)    
def Processing(A, directory2, directory3, a, fiction, BB, CC,directory4, directory5):
    while True:
        try:
            TU3 = []
            VP3 =[]
            feed = gtfs_realtime_pb2.FeedMessage()
            try:
                response = urllib.request.urlopen('https://gtfsrt.api.translink.com.au/Feed/SEQ')
                feed.ParseFromString(response.read())
            except:
                continue
            for entity in feed.entity:
                if entity.HasField('trip_update'):
                    TU3.append(str(entity.trip_update))
            for entity in feed.entity:
                if entity.HasField('vehicle'):
                    VP3.append(str(entity.vehicle))
            A = TU3
            t = float(time.time())
            tt = 'VP'
            createCSVfileWCD(A, directory2 + "/"  +a+str(t)+'_'+'.csv')
            createCSVfileWCD(VP3, directory5 + "/"  +tt+str(t)+'_'+'.csv') #'Trip_Update'
            A = inputCSVfile(directory2 + "/"  +a+str(t)+'_'+'.csv') #'Trip_Update'
            B = ''.join(str(e) for e in A)
            FO = []
            for matchedtext in re.findall(r'(?<=trip {).*?(?<=timestamp: \d\d\d\d\d\d\d\d\d\d)', B):
                #print(matchedtext)
                #matchedtext = ''.join(str(e) for e in matchedtext)
                tripid = re.findall(r'(?<=trip_id: ").*?(?=")', matchedtext)
                Start_time = re.findall(r'(?<=start_time: ").*?(?=")', matchedtext)
                Start_date = re.findall(r'(?<=start_date: ").*?(?=")', matchedtext)
                schedule_relationship = re.findall(r"(?<=schedule_relationship:).*?(?=')", matchedtext)
                if len(schedule_relationship)!=0:
                    sr = schedule_relationship[0]
                else:
                    sr = '0'
                #print(schedule_relationship, 'schedule_relationship.................')
                route_id = re.findall(r'(?<=route_id: ").*?(?=")', matchedtext)
                Vehid = re.findall(r"(?<='id: ).*?(?=')", matchedtext)
                timestamp = matchedtext[-10:]
                for stopseq in re.findall(r'(?<=stop_time_update {).*?(?<=\d")',matchedtext ):
                    #print(stopseq)
                    try:
                        Stop_Seq = re.findall(r"(?<=stop_sequence: ).*?(?=')", stopseq)
                        #print(Stop_Seq)
                        arrival = re.findall(r"(?<=arrival {').*?(?=})", stopseq)
                        arrival = ''.join(str(e) for e in arrival)
                        Arrival_time = re.findall(r"(?<='time: ).*?(?=')", arrival)
                        #print(Arrival_time, 'Arrival_time....................................')
                    except:
                        Stop_Seq = ''
                        pass
                    try:
                        att = Arrival_time[0]
                        #print(att, 'attttttttttttt')
                        at = Brisbane(float(Arrival_time[0]))
                        #print(at)
                    except:
                        att = '0'
                        at = ''
                        pass
                    Arrival_Uncertainity = re.findall(r"(?<=uncertainty: ).*?(?=')", arrival)
                    try:
                        au = Arrival_Uncertainity[0]
                    except:
                        au = ''
                        pass
                    Arrival_Delay = re.findall(r"(?<=delay: ).*?(?=')", arrival)
                    try:
                        ad = Arrival_Delay[0]
                    except:
                        ad = '0'
                        pass
                    departure = re.findall(r"(?<=departure {').*?(?=})", stopseq)
                    departure = ''.join(str(e) for e in departure)
                    departure_time = re.findall(r"(?<='time: ).*?(?=')", departure)
                    try:
                        dtt = departure_time[0]
                        dt = Brisbane(float(departure_time[0]))
                    except:
                        dtt = '0'
                        dt = ''
                        pass
                    departure_Uncertainity = re.findall(r"(?<=uncertainty: ).*?(?=')", departure)
                    try:
                        du =  departure_Uncertainity[0]
                    except:
                        du = ''
                        pass
                    try:
                        sid = re.findall(r'(?<=stop_id: ").*?(?=")', stopseq)
                    except:
                        sid = ''
                        pass

                    try:

                        if float(timestamp) > float(att)-300:
                            data = [ tripid[0],Start_time[0],Start_date[0],route_id[0],Stop_Seq[0], float(ad) , float(att), at, au , float(dtt), dt, du, sid[0],sr,Vehid[0],timestamp]
                            #print(data)
                            FO.append(data)
                        
                    
                    except:
                        if len(Stop_Seq)>0:
                            if Stop_Seq[0] =='':
                                data = [ tripid[0],Start_time[0],Start_date[0],route_id[0],Stop_Seq[0], float(ad) , float(att), at, au , float(dtt), dt, du, sid[0],sr,Vehid[0],timestamp]
                                FO.append(data)
                        pass
            Dt = pd.DataFrame(FO, columns=[ "Trip ID" , "Star t Time", "Start Date", "Route_ID", "Stop Sequence","Arrival Delay", "Arrival Time (UNIX)","Arrival Time (UTC+10)", "Arrival Uncertainty", "Departure Time (UNIX)","Departure Time (UTC+10)", "Departure Uncertainty", "stop_id","schedule_relationship" ,"Vehicle ID", "timestamp" ])
            Dt.Route_ID.apply(str)
            Dt.to_csv(directory3 + '/' + "Table TU"+str(t)+'_'+'.csv', index = None, header=True)
            SpeedProcessing(Dt,directory4 , BB, CC, str(t))
            response.close()
            break
        except:
            break
        

#Input Folder Names
TU = "TripUpdate entity"
VP = "VehiclePosition entity"
SA = "ServiceAlert entity"
FT = "File Tracker"

TU1directory= Working_Directory+ 'TripUpdate entity'
if not os.path.exists(TU1directory):
    os.makedirs(TU1directory)

VP1directory= Working_Directory+ 'VehiclePosition entity'
if not os.path.exists(VP1directory):
    os.makedirs(VP1directory)

SA1directory= Working_Directory+ 'ServiceAlert entity'
if not os.path.exists(SA1directory):
    os.makedirs(SA1directory)

FTdirectory= Working_Directory+ 'File Tracker'
if not os.path.exists(FTdirectory):
    os.makedirs(FTdirectory)
                                        

t = int(time.time())
tm1 = float(t)
tm2 = float(t)

datee = datetime.datetime.strptime(str(Brisbane(t)),"%d-%m-%Y %H:%M:%S" )
print(datee)
m = datee.month
y = datee.year
d = datee.day
d1 = d
m_name = monthname(Brisbane(t))

ttt = str(d)+"-"+str(m)+"-"+str(y)
ttt1 = str(m_name)+" ,"+str(y)
FTdirectory= FTdirectory+ "/"+ 'FT ' +str(ttt1) 
FTdirectory2= FTdirectory+ "/"+ 'FT - TripUpdate '+str(ttt1) + "/"
FTdirectory3= FTdirectory+ "/"+ 'FT - VehiclePositions '+str(ttt1) +"/"
FTdirectory4= FTdirectory+ "/"+ 'FT - Alert '+str(ttt1)+"/"
try:
    File_Tracker_TU1= inputCSVfile(FTdirectory2+'File_Tracker_TU '+str(d)+"-"+ str(m)+"-"+str(y)+ " " +'.csv')
except:
    File_Tracker_TU1 =[]
try:
    File_Tracker_VP1= inputCSVfile(FTdirectory3+'File_Tracker_VP '+str(d)+"-"+ str(m)+"-"+str(y)+ " " +'.csv')
except:
    File_Tracker_VP1 =[]
try:
    File_Tracker_SA1 = inputCSVfile(FTdirectory4+'File_Tracker_SA '+str(d)+"-"+ str(m)+"-"+str(y)+ " " +'.csv')
except:
    File_Tracker_SA1 =[]


ii = 0
A =[]
B =[]
C = []


t2 = 0
t1= float(time.time())
Trip_Update = []
Vehicle_Positions = []
VP4 =[]
TU4 =[]
fiction = ['0','0','0','0']
BB = pd.read_csv(Working_Directory+'Route Shape Stop.csv', header=0)
CC = pd.read_csv(Working_Directory+'Tripwise- Stop wise distances.csv', header=0)
if __name__ == "__main__":
    while True:
        try:
            while True:
                t = int(time.time())
                datee = datetime.datetime.strptime(str(Brisbane(t)),"%d-%m-%Y %H:%M:%S" )
                m = datee.month
                y = datee.year
                d = datee.day
                
                m_name = monthname(Brisbane(t))
                if t2!=0:
                    if d!=d1:
                        File_Tracker_TU1 =[]
                        File_Tracker_VP1 =[]
                        File_Tracker_SA1 =[]
                d1 = d
                t2 = 1
                ii= ii+1
                ttt = str(d)+"-"+str(m)+"-"+str(y)
                ttt1 = str(m_name)+" ,"+str(y)
                feed = gtfs_realtime_pb2.FeedMessage()
                ii= ii+1
                TUdirectory = Working_Directory + TU +"/" + 'TU '+str(ttt1)
                VPdirectory = Working_Directory + VP +"/" + 'VP '+str(ttt1)
                SAdirectory = Working_Directory + SA +"/" + 'SA '+str(ttt1)
                if not os.path.exists(TUdirectory):
                    os.makedirs(TUdirectory)
                TUdirectory2= TUdirectory+ "/"+ 'TU '+str(ttt)
                if not os.path.exists(TUdirectory2):
                    os.makedirs(TUdirectory2)               
                if not os.path.exists(VPdirectory):
                    os.makedirs(VPdirectory)
                VPdirectory2= VPdirectory+ "/"+ 'VP '+str(ttt)
                if not os.path.exists(VPdirectory2):
                    os.makedirs(VPdirectory2)
                    
                TUdirectory20= TUdirectory+ "/"+ 'TU '+str(d)
                VPdirectory20= VPdirectory+ "/"+ 'VP '+str(d)
                if not os.path.exists(TUdirectory20):
                    os.makedirs(TUdirectory20)
                if not os.path.exists(VPdirectory20):
                    os.makedirs(VPdirectory20)

                TUdirectory30= TUdirectory+ "/"+ 'Speed TU '+str(d)

                if not os.path.exists(TUdirectory30):
                    os.makedirs(TUdirectory30)


                SAdirectory = Working_Directory + SA +"/" + 'SA '+str(ttt1)
                if not os.path.exists(SAdirectory):
                    os.makedirs(SAdirectory)
                SAdirectory2= SAdirectory+ "/"+ 'SA '+str(ttt)
                if not os.path.exists(SAdirectory2):
                    os.makedirs(SAdirectory2)

                FTdirectory = Working_Directory + FT +"/" + 'FT '+str(ttt1)
                if not os.path.exists(FTdirectory):
                    os.makedirs(FTdirectory)
                    
                    
                FTdirectory2= FTdirectory+ "/"+ 'FT - TripUpdate '+str(ttt1) + "/"
                FTdirectory3= FTdirectory+ "/"+ 'FT - VehiclePositions '+str(ttt1) +"/"
                FTdirectory4= FTdirectory+ "/"+ 'FT - Alert '+str(ttt1)+"/"
                if not os.path.exists(FTdirectory2):
                    os.makedirs(FTdirectory2)
                    File_Tracker_TU1 =[]
                if not os.path.exists(FTdirectory3):
                    os.makedirs(FTdirectory3)
                    File_Tracker_VP1 =[]
                if not os.path.exists(FTdirectory3):
                    os.makedirs(FTdirectory3)
                    File_Tracker_SA1 =[]

                
                VP3 = []
                TU3 = []

                hhhh = 'TU'
                Trip_Update = [['a', 'b'], ['c', 'd'], ['c', 'd']]
                try:
                    if float(time.time()) - t1 >= 8:
                        hhhh = str('Trip_Update')
                        for n in range(0, len(Trip_Update)-2):
                            p2 = multiprocessing.Process(target = Processing, args = (Trip_Update[n], TUdirectory2, TUdirectory20, hhhh, fiction, BB, CC, TUdirectory30,VPdirectory2,))
                            p2.start()
                            print('started 1', (time.time() - t1))
                        t1 = time.time()
                        Trip_Update = []
                        TU4 =[]
                        break

                except:
                    print('fir se error')
                    pass
        except:
            print('eeee')
            continue

#from notify_run import Notify
#import time
#notify = Notify()

#stopped_at = time.ctime()
#notify.send("GTFS code stopped at : " + stopped_at)





