import mmap
import struct
import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _convert_degree_arc_to_percent(value):
    """Converte gradi d'arco in percentuale (0..1)."""
    return max(value / 360.0, 0.0)


# ---------------------------------------------------------------------------
# Costanti Shared Memory Assetto Corsa
# ---------------------------------------------------------------------------

_SM_PHYSICS   = "Local\\acpmf_physics"
_SM_GRAPHICS  = "Local\\acpmf_graphics"
_SM_STATIC    = "Local\\AcTools.Static"

# Offset nella SM graphics per posizione X, Y, Z della telecamera (proxy posizione auto)
# Layout: vedi documentazione AC shared memory
# I campi rilevanti in acpmf_graphics partono con: packetId(i), status(i), session(i),
# currentTime / lastTime / bestTime (3x 15 char unicode = 30 bytes ciascuna) ...
# La posizione cameraPosition è a offset 716 in float[3]: X, Y, Z
_GRAPHICS_POS_OFFSET = 716   # byte offset per cameraPosition[3] (X, Y, Z) — float


# ---------------------------------------------------------------------------
# Lettura Track Name da AC_SM_Static
# ---------------------------------------------------------------------------

def get_track_name(fallback: str = "monza") -> str:
    """
    Legge la Shared Memory Static di Assetto Corsa per ottenere il nome della pista.
    Ritorna il fallback se la SM non è disponibile.
    """
    try:
        shm = mmap.mmap(-1, 512, _SM_STATIC, access=mmap.ACCESS_READ)
        # Track name in SM Static è una stringa Unicode (UTF-16-LE) di max 30 char
        # a partire dall'offset 20 (dopo il campo smVersion e acVersion)
        raw = shm[20:80]
        shm.close()
        # Split sul primo null-word e decodifica
        name = raw.split(b'\x00\x00')[0].decode('utf-16-le', errors='ignore').strip()
        return name if name else fallback
    except Exception:
        return fallback


# ---------------------------------------------------------------------------
# Parsing file AI line (.ai) di Assetto Corsa
# ---------------------------------------------------------------------------

def load_ai_line(filepath: str) -> np.ndarray:
    """
    Legge il file binario fast_lane.ai di AC e restituisce un array numpy (N, 7).
    Ogni riga contiene: [x, y, z, speed, ?, ?, ?].
    """
    points = []
    with open(filepath, "rb") as f:
        num_points = struct.unpack('i', f.read(4))[0]
        for _ in range(num_points):
            data = struct.unpack('fffffff', f.read(28))
            points.append(data)
    return np.array(points, dtype=np.float32)


def build_kdtree(ai_line: np.ndarray) -> KDTree:
    """
    Costruisce un KDTree 2D (X, Z) dalla linea AI per query di distanza rapide.
    """
    return KDTree(ai_line[:, [0, 2]])


# ---------------------------------------------------------------------------
# Lettura posizione auto da acpmf_graphics
# ---------------------------------------------------------------------------

def get_car_position() -> tuple[float, float]:
    """
    Legge X, Z della posizione auto dalla Shared Memory Graphics di AC.
    Ritorna (0.0, 0.0) se la SM non è disponibile.
    """
    try:
        shm = mmap.mmap(-1, _GRAPHICS_POS_OFFSET + 12, _SM_GRAPHICS, access=mmap.ACCESS_READ)
        shm.seek(_GRAPHICS_POS_OFFSET)
        x, y, z = struct.unpack('fff', shm.read(12))
        shm.close()
        return float(x), float(z)
    except Exception:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Driver principale: lettura telemetria fisica in tempo reale
# ---------------------------------------------------------------------------

class AssettoCorsaData:
    """
    Legge la memoria condivisa acpmf_physics di Assetto Corsa e
    espone tutti i campi telemetrici come attributi (es. self.rpm, self.speed).
    """

    # Nomi campi — stringa originale testata (NON modificare l'ordine)
    FIELDS = (
        'packetId throttle brake fuel gear rpm steerAngle speed velocity1 velocity2 velocity3 '
        'accGX accGY accGZ '
        'wheelSlipFL wheelSlipFR wheelSlipRL wheelSlipRR '
        'wheelLoadFL wheelLoadFR wheelLoadRL wheelLoadRR '
        'wheelsPressureFL wheelsPressureFR wheelsPressureRL wheelsPressureRR '
        'wheelAngularSpeedFL wheelAngularSpeedFR wheelAngularSpeedRL wheelAngularSpeedRR '
        'TyrewearFL TyrewearFR TyrewearRL TyrewearRR '
        'tyreDirtyLevelFL tyreDirtyLevelFR tyreDirtyLevelRL tyreDirtyLevelRR '
        'TyreCoreTempFL TyreCoreTempFR TyreCoreTempRL TyreCoreTempRR '
        'camberRADFL camberRADFR camberRADRL camberRADRR '
        'suspensionTravelFL suspensionTravelFR suspensionTravelRL suspensionTravelRR '
        'drs tc1 heading pitch roll cgHeight '
        'carDamagefront carDamagerear carDamageleft carDamageright carDamagecentre '
        'numberOfTyresOut pitLimiterOn abs1 '
        'kersCharge kersInput automat rideHeightfront rideHeightrear turboBoost ballast '
        'airDensity airTemp roadTemp '
        'localAngularVelX localAngularVelY localAngularVelZ '
        'finalFF performanceMeter engineBrake '
        'ersRecoveryLevel ersPowerLevel ersHeatCharging ersIsCharging kersCurrentKJ '
        'drsAvailable drsEnabled '
        'brakeTempFL brakeTempFR brakeTempRL brakeTempRR clutch '
        'tyreTempI1 tyreTempI2 tyreTempI3 tyreTempI4 '
        'tyreTempM1 tyreTempM2 tyreTempM3 tyreTempM4 '
        'tyreTempO1 tyreTempO2 tyreTempO3 tyreTempO4 '
        'isAIControlled '
        'tyreContactPointFLX tyreContactPointFLY tyreContactPointFLZ '
        'tyreContactPointFRX tyreContactPointFRY tyreContactPointFRZ '
        'tyreContactPointRLX tyreContactPointRLY tyreContactPointRLZ '
        'tyreContactPointRRX tyreContactPointRRY tyreContactPointRRZ '
        'tyreContactNormalFLX tyreContactNormalFLY tyreContactNormalFLZ '
        'tyreContactNormalFRX tyreContactNormalFRY tyreContactNormalFRZ '
        'tyreContactNormalRLX tyreContactNormalRLY tyreContactNormalRLZ '
        'tyreContactNormalRRX tyreContactNormalRRY tyreContactNormalRRZ '
        'tyreContactHeadingFLX tyreContactHeadingFLY tyreContactHeadingFLZ '
        'tyreContactHeadingFRX tyreContactHeadingFRY tyreContactHeadingFRZ '
        'tyreContactHeadingRLX tyreContactHeadingRLY tyreContactHeadingRLZ '
        'tyreContactHeadingRRX tyreContactHeadingRRY tyreContactHeadingRRZ '
        'brakeBias localVelocityX localVelocityY localVelocityZ '
        'P2PActivation P2PStatus currentMaxRpm '
        'mz1 mz2 mz3 mz4 fx1 fx2 fx3 fx4 fy1 fy2 fy3 fy4 '
        'slipRatio1 slipRatio2 slipRatio3 slipRatio4 '
        'slipAngle1 slipAngle2 slipAngle3 slipAngle4 '
        'tcinAction absInAction '
        'suspensionDamage1 suspensionDamage2 suspensionDamage3 suspensionDamage4 '
        'tyreTemp1 tyreTemp2 tyreTemp3 tyreTemp4 waterTemp '
        'brakePressureFL brakePressureFR brakePressureRL brakePressureRR '
        'frontBrakeCompound rearBrakeCompound '
        'padLifeFL padLifeFR padLifeRL padLifeRR '
        'discLifeFL discLifeFR discLifeRL discLifeRR'
    ).split()

    # Layout struct originale testato — corrisponde 1:1 ai FIELDS sopra
    _LAYOUT = 'ifffiiffffffff 4f fffffffffffffffffffffffffffffffffffffffffffiifffiffffffffffffiiiiifiifffffffffffffffffiffffffffffffffffffffffffffffffffffffffffiifffffffffffffffffffffiifffffffffffffiiffffffff'

    # Gruppi di campi da raggruppare in liste [FL, FR, RL, RR]
    _ARRAY_GROUPS = [
        'wheelSlip', 'wheelLoad', 'wheelsPressure', 'wheelAngularSpeed',
        'Tyrewear', 'tyreDirtyLevel', 'TyreCoreTemp', 'camberRAD',
        'suspensionTravel', 'brakeTemp', 'brakePressure', 'padLife', 'discLife',
    ]

    def __init__(self):
        print('[Driver] Inizializzazione AssettoCorsaData...')
        self._fields = self.FIELDS
        self._size = struct.calcsize(self._LAYOUT)
        self._mmap = None

        # Pre-inizializza tutti gli attributi a 0
        for field in self._fields:
            setattr(self, field, 0.0)

    # ------------------------------------------------------------------
    # Connessione
    # ------------------------------------------------------------------

    def start(self):
        """Apre la connessione alla Shared Memory acpmf_physics."""
        print('[Driver] Connessione alla physics SM...')
        if not self._mmap:
            self._mmap = mmap.mmap(-1, self._size, _SM_PHYSICS, access=mmap.ACCESS_READ)

    def stop(self):
        """Chiude la connessione alla Shared Memory."""
        print('[Driver] Chiusura connessione SM...')
        if self._mmap:
            self._mmap.close()
        self._mmap = None

    # ------------------------------------------------------------------
    # Aggiornamento dati
    # ------------------------------------------------------------------

    def update(self):
        """
        Legge la SM physics e aggiorna tutti gli attributi.
        Dopo la chiamata è possibile usare: self.rpm, self.speed, self.numberOfTyresOut, ecc.
        """
        if not self._mmap:
            return

        self._mmap.seek(0)
        raw = self._mmap.read(self._size)
        unpacked = struct.unpack(self._LAYOUT, raw)

        data = {self._fields[i]: v for i, v in enumerate(unpacked)}

        # Raggruppa i campi FL/FR/RL/RR in liste
        for group in self._ARRAY_GROUPS:
            data[group] = []
            for suffix in ('FL', 'FR', 'RL', 'RR'):
                key = group + suffix
                if key in data:
                    data[group].append(_convert_degree_arc_to_percent(data.pop(key)))

        # Imposta attributi dinamicamente
        for key, value in data.items():
            setattr(self, key, value)

    # ------------------------------------------------------------------
    # Proprietà di comodo
    # ------------------------------------------------------------------

    @property
    def tyres_out(self) -> int:
        """Numero di ruote fuori dalla pista (0..4)."""
        return int(self.numberOfTyresOut)

    @property
    def speed_ms(self) -> float:
        """Velocità in m/s (velocità AC è in km/h)."""
        return self.speed / 3.6


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    print(f"Pista corrente: {get_track_name()}")

    reader = AssettoCorsaData()
    reader.start()
    try:
        while True:
            reader.update()
            x, z = get_car_position()
            print(
                f"RPM: {reader.rpm:6.0f} | "
                f"Speed: {reader.speed:6.1f} km/h | "
                f"Gear: {reader.gear} | "
                f"TyresOut: {reader.tyres_out} | "
                f"Pos: ({x:.1f}, {z:.1f})"
            )
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()