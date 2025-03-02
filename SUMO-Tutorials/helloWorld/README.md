# Hello World Tutorial for SUMO

## Useful Links
- [Hello World Tutorial](https://sumo.dlr.de/docs/Tutorials/Hello_World.html)
- [Hello SUMO Tutorial](https://sumo.dlr.de/docs/Tutorials/Hello_SUMO.html)
- [Official Source From SUMO](https://github.com/eclipse-sumo/sumo/tree/main/tests/complex/tutorial/hello)

## Requirements
- **SUMO GUI** and **netedit** version **≥ 1.4.0**

---
## GUI-based Approach
### Basic Files for a SUMO Simulation
| File Type        | Extension         | Description |
|-----------------|------------------|-------------|
| **Network File** | `.net.xml`       | Defines the road network |
| **Route File**   | `.rou.xml`       | Contains vehicle routes |
| **Configuration File** | `.sumocfg` | SUMO simulation configuration |

---
### Using netedit
#### **Creating and Editing a Network**
- **Create a new network**: `Ctrl + N` (or File → New Network)
- **Enter Edge Mode**: `E` (or Edit → Edge Mode)
- **Enter Inspect Mode**: `I` (or Edit → Inspect Mode)
- **Save Network**: `Ctrl + S` (or File → Save Network)
- **Save Network As**: `Ctrl + Shift + S` (or File → Save Network As)

#### **Defining Traffic Demand**
- **Enter Route Mode**: `R` (or Edit → Route Mode)
- **Finish Route Creation**: Click `Compose Route` Button
- **Enter Vehicle Mode**: `V` (or Edit → Vehicle Mode)
- **Save Demand Elements**: `Ctrl + Shift + D` (or File → Demand Elements → Save Demand Elements)
- **Save Demand Elements As**: File → Demand Elements As

---
### Using sumo-gui
- **Open SUMO-GUI from netedit**: `Ctrl + T`
- **Save Configuration As**: `Ctrl + Shift + S` (or File → Save Configuration)
- **Run Simulation**: `Ctrl + A`
- **Reload Simulation**: Click `Reload` Button

---
## Script-based Approach
### **Generating a Network from Nodes and Edges**
```sh
netconvert --node-files=helloWorld.nod.xml --edge-files=helloWorld.edg.xml --output-file=helloWorld.net.xml
```

### **Starting the Simulation**
```sh
sumo -c hello.sumocfg  # Command-line mode
sumo-gui -c hello.sumocfg  # GUI mode
```

### **Workflow**

1️⃣ **Create Nodes File**

2️⃣ **Create Edges File**

3️⃣ **Generate Network File** (from Nodes and Edges)

4️⃣ **Create Routes File**

5️⃣ **Create SUMO Configuration File**

6️⃣ **Start Simulation**

7️⃣ **Create ViewSettings File** (and add it to the SUMO Configuration File)

### **Advanced Resources**
- **[Defining Networks using XML](https://sumo.dlr.de/docs/Networks/PlainXML.html)**
- **[Using netconvert](https://sumo.dlr.de/docs/netconvert.html)**
  - Advanced Import: [Import Networks](https://sumo.dlr.de/docs/Networks/Import.html)
- **[Defining Vehicles and Routes](https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html)**

---
