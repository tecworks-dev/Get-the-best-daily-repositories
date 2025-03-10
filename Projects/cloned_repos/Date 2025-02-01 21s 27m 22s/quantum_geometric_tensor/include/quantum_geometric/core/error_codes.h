#ifndef QUANTUM_ERROR_CODES_H
#define QUANTUM_ERROR_CODES_H

// Error codes
typedef enum {
    QGT_SUCCESS = 0,
    QGT_ERROR_INVALID_PARAMETER = -1,
    QGT_ERROR_INVALID_ARGUMENT = QGT_ERROR_INVALID_PARAMETER,  // Alias for backward compatibility
    QGT_ERROR_MEMORY_ALLOCATION = -2,
    QGT_ERROR_DIMENSION_MISMATCH = -3,
    QGT_ERROR_INVALID_STATE = -4,
    QGT_ERROR_VALIDATION_FAILED = -5,
    QGT_ERROR_HARDWARE_FAILURE = -6,
    QGT_ERROR_NOT_IMPLEMENTED = -7,
    QGT_ERROR_OVERFLOW = -8,
    QGT_ERROR_UNDERFLOW = -9,
    QGT_ERROR_DIVISION_BY_ZERO = -10,
    QGT_ERROR_INVALID_OPERATION = -11,
    QGT_ERROR_SYSTEM_FAILURE = -12,
    QGT_ERROR_TIMEOUT = -13,
    QGT_ERROR_RESOURCE_EXHAUSTED = -14,
    QGT_ERROR_INTERNAL = -15,
    QGT_ERROR_IO_ERROR = -16,
    QGT_ERROR_ALREADY_INITIALIZED = -17,
    QGT_ERROR_NOT_INITIALIZED = -18,
    QGT_ERROR_INCOMPATIBLE = -19,
    QGT_ERROR_INVALID_METRIC = -20,
    QGT_ERROR_INVALID_CURVATURE = -21,
    QGT_ERROR_INVALID_DIMENSION = -22,
    QGT_ERROR_NO_MEMORY = -23,
    QGT_ERROR_NUMERICAL_INSTABILITY = -24,
    QGT_ERROR_INVALID_PROPERTY = -25,
    QGT_ERROR_INITIALIZATION = -26,
    QGT_ERROR_RESOURCE_UNAVAILABLE = -27,
    QGT_ERROR_BUFFER_OVERFLOW = -28,
    QGT_ERROR_INCOMPATIBLE_TYPES = -29,
    QGT_ERROR_INVALID_TENSOR = -30,
    QGT_ERROR_INVALID_HARDWARE = -31,
    QGT_ERROR_INVALID_CONFIG = -32,
    QGT_ERROR_NOT_HERMITIAN = -33,
    QGT_ERROR_NOT_POSITIVE_DEFINITE = -34,
    QGT_ERROR_INVALID_CACHE = -33,
    QGT_ERROR_INVALID_CIRCUIT = -34,
    QGT_ERROR_INVALID_SYSTEM = -35,
    QGT_ERROR_INVALID_OPTIMIZATION = -36,
    QGT_ERROR_INVALID_MEMORY = -37,
    QGT_ERROR_INVALID_PROTECTION = -38,
    QGT_ERROR_INVALID_DISTRIBUTION = -39,
    QGT_ERROR_INVALID_WORKLOAD = -40,
    QGT_ERROR_INVALID_PROCESS = -41,
    QGT_ERROR_INVALID_THREAD = -42,
    QGT_ERROR_INVALID_TASK = -43,
    QGT_ERROR_INVALID_QUEUE = -44,
    QGT_ERROR_INVALID_SCHEDULER = -45,
    QGT_ERROR_INVALID_ALIGNMENT = -46,
    QGT_ERROR_NOT_SUPPORTED = -47,
    QGT_ERROR_SVD_FAILED = -48,
    QGT_ERROR_EIGENDECOMPOSITION_FAILED = -49,
    QGT_ERROR_QR_FAILED = -50,
    QGT_ERROR_LU_FAILED = -51,
    QGT_ERROR_MATRIX_SINGULAR = -52,
    QGT_ERROR_INVERSION_FAILED = -53,

    // Hardware backend errors
    QGT_ERROR_HARDWARE_NOT_AVAILABLE = -100,
    QGT_ERROR_HARDWARE_BUSY = -101,
    QGT_ERROR_HARDWARE_CALIBRATION = -102,
    QGT_ERROR_HARDWARE_CONNECTIVITY = -103,
    QGT_ERROR_HARDWARE_COHERENCE = -104,
    QGT_ERROR_HARDWARE_GATE = -105,
    QGT_ERROR_HARDWARE_READOUT = -106,
    QGT_ERROR_HARDWARE_RESET = -107,
    QGT_ERROR_HARDWARE_QUEUE_FULL = -108,
    QGT_ERROR_HARDWARE_COMMUNICATION = -109,
    QGT_ERROR_HARDWARE_AUTHENTICATION = -110,
    QGT_ERROR_HARDWARE_CONFIGURATION = -111,
    QGT_ERROR_NO_GPU_BACKEND = -112,
    QGT_ERROR_NO_METAL_BACKEND = -113,
    QGT_ERROR_NO_CUDA_BACKEND = -114,
    QGT_ERROR_UNSUPPORTED_FEATURE = -115,

    // Simulator errors
    QGT_ERROR_SIMULATOR_MEMORY = -200,
    QGT_ERROR_SIMULATOR_PRECISION = -201,
    QGT_ERROR_SIMULATOR_STATE = -202,
    QGT_ERROR_SIMULATOR_OPERATION = -203,
    QGT_ERROR_SIMULATOR_MEASUREMENT = -204,
    QGT_ERROR_SIMULATOR_INITIALIZATION = -205,
    QGT_ERROR_SIMULATOR_DECOHERENCE = -206,
    QGT_ERROR_SIMULATOR_NOISE = -207,
    QGT_ERROR_SIMULATOR_VALIDATION = -208,

    // Program/circuit errors
    QGT_ERROR_PROGRAM_INVALID = -300,
    QGT_ERROR_PROGRAM_SYNTAX = -301,
    QGT_ERROR_PROGRAM_QUBITS = -302,
    QGT_ERROR_PROGRAM_GATES = -303,
    QGT_ERROR_PROGRAM_MEASUREMENT = -304,
    QGT_ERROR_PROGRAM_COMPILATION = -305,
    QGT_ERROR_PROGRAM_OPTIMIZATION = -306,
    QGT_ERROR_PROGRAM_VALIDATION = -307,
    QGT_ERROR_PROGRAM_EXECUTION = -308,
    QGT_ERROR_PROGRAM_TIMEOUT = -309,

    // Error mitigation errors
    QGT_ERROR_MITIGATION_FAILED = -400,
    QGT_ERROR_MITIGATION_CALIBRATION = -401,
    QGT_ERROR_MITIGATION_MODEL = -402,
    QGT_ERROR_MITIGATION_SAMPLING = -403,
    QGT_ERROR_MITIGATION_EXTRAPOLATION = -404,
    QGT_ERROR_MITIGATION_VALIDATION = -405,
    QGT_ERROR_MITIGATION_RESOURCE = -406,
    QGT_ERROR_MITIGATION_DATA = -407,
    QGT_ERROR_MITIGATION_CONVERGENCE = -408,

    // Communication errors
    QGT_ERROR_COMMUNICATION_FAILED = -500,
    QGT_ERROR_COMMUNICATION_TIMEOUT = -501,
    QGT_ERROR_COMMUNICATION_NETWORK = -502,
    QGT_ERROR_COMMUNICATION_PROTOCOL = -503,
    QGT_ERROR_COMMUNICATION_ENCRYPTION = -504,
    QGT_ERROR_COMMUNICATION_DATA = -505,
    QGT_ERROR_COMMUNICATION_STATE = -506,

    // Authentication errors
    QGT_ERROR_AUTH_FAILED = -600,
    QGT_ERROR_AUTH_TOKEN = -601,
    QGT_ERROR_AUTH_CREDENTIALS = -602,
    QGT_ERROR_AUTH_PERMISSION = -603,
    QGT_ERROR_AUTH_EXPIRED = -604,
    QGT_ERROR_AUTH_RATE_LIMIT = -605,
    QGT_ERROR_AUTH_CONFIG = -606,

    QGT_ERROR_ALLOCATION_FAILED = QGT_ERROR_MEMORY_ALLOCATION  // Alias for backward compatibility
} qgt_error_t;

#endif // QUANTUM_ERROR_CODES_H
