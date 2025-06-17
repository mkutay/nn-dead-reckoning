from datetime import datetime

class SingularData:
    path: str
    timestamp: datetime
    index: int
    lat: float # latitude of the oxts-unit (deg)
    lon: float # longitude of the oxts-unit (deg)
    alt: float # altitude of the oxts-unit (m)
    roll: float # roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
    pitch: float # pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
    yaw: float # heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
    vn: float # velocity towards north (m/s)
    ve: float # velocity towards east (m/s)
    vf: float # forward velocity, i.e. parallel to earth-surface (m/s)
    vl: float # leftward velocity, i.e. parallel to earth-surface (m/s)
    vu: float # upward velocity, i.e. perpendicular to earth-surface (m/s)
    ax: float # acceleration in x, i.e. in direction of vehicle front (m/s^2)
    ay: float # acceleration in y, i.e. in direction of vehicle left (m/s^2)
    az: float # acceleration in z, i.e. in direction of vehicle top (m/s^2)
    af: float # forward acceleration (m/s^2)
    al: float # leftward acceleration (m/s^2)
    au: float # upward acceleration (m/s^2)
    wx: float # angular rate around x (rad/s)
    wy: float # angular rate around y (rad/s)
    wz: float # angular rate around z (rad/s)
    wf: float # angular rate around forward axis (rad/s)
    wl: float # angular rate around leftward axis (rad/s)
    wu: float # angular rate around upward axis (rad/s)
    pos_accuracy: float # velocity accuracy (north/east in m)
    vel_accuracy: float # velocity accuracy (north/east in m/s)
    navstat: int # navigation status (see navstat_to_string)
    numsats: int # number of satellites tracked by primary GPS receiver
    posmode: int # position mode of primary GPS receiver (see gps_mode_to_string)
    velmode: int # velocity mode of primary GPS receiver (see gps_mode_to_string)
    orimode: int # orientation mode of primary GPS receiver (see gps_mode_to_string)

    def __init__(self, path: str, timestamp: datetime, index: int) -> None:
        self.path = path
        self.timestamp = timestamp
        self.index = index
        self.read_data()

    def read_data(self) -> None:
        full_path = f"{self.path}/{self.index:010d}.txt"
        with open(full_path, 'r') as file:
            data = file.read().split()
            if len(data) != 30:
                raise ValueError(f"Expected 30 data fields, got {len(data)} in {self.path}/{self.index:010d}.txt")
            self.lat = float(data[0])
            self.lon = float(data[1])
            self.alt = float(data[2])
            self.roll = float(data[3])
            self.pitch = float(data[4])
            self.yaw = float(data[5])
            self.vn = float(data[6])
            self.ve = float(data[7])
            self.vf = float(data[8])
            self.vl = float(data[9])
            self.vu = float(data[10])
            self.ax = float(data[11])
            self.ay = float(data[12])
            self.az = float(data[13])
            self.af = float(data[14])
            self.al = float(data[15])
            self.au = float(data[16])
            self.wx = float(data[17])
            self.wy = float(data[18])
            self.wz = float(data[19])
            self.wf = float(data[20])
            self.wl = float(data[21])
            self.wu = float(data[22])
            self.pos_accuracy = float(data[23])
            self.vel_accuracy = float(data[24])
            self.navstat = int(float(data[25]))
            self.numsats = int(float(data[26]))
            self.posmode = int(float(data[27]))
            self.velmode = int(float(data[28]))
            self.orimode = int(float(data[29]))