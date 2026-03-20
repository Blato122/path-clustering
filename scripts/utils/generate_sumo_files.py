from idf_utils import *

OSM_FILE = "../data/andorra/andorra.osm"
NET_FILE = "../data/andorra/andorra.net.xml"
ROU_FILE = "../data/andorra/andorra.rou.xml"
NET_NAME = "andorra"

if not os.path.exists(OSM_FILE):
    raise FileNotFoundError(f"OSM file '{OSM_FILE}' not found.")
print("Found OSM file:", OSM_FILE)

convert_osm_to_net(OSM_FILE, NET_FILE) # OSM -> net.xml
create_sumo_miscellaneous(NET_NAME, NET_FILE) # net.xml -> [edg, con, nod, tll, typ].xml
convert_net_to_rou(NET_FILE, ROU_FILE) # net.xml -> rou.xml
