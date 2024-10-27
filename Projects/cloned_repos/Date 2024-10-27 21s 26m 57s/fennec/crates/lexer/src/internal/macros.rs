macro_rules! start_of_identifier {
    () => {
        b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'\x80'..=b'\xff'
    };
}

macro_rules! part_of_identifier {
    () => {
        b'0'..=b'9' | b'a'..=b'z' | b'A'..=b'Z' | b'_' | b'\x80'..=b'\xff'
    };
}

macro_rules! start_of_number {
    () => {
        b'0'..=b'9'
    };
}

macro_rules! start_of_binary_number {
    () => {
        [b'0', b'B' | b'b']
    };
}

macro_rules! start_of_octal_number {
    () => {
        [b'0', b'O' | b'o']
    };
}

macro_rules! start_of_hexadecimal_number {
    () => {
        [b'0', b'X' | b'x']
    };
}

macro_rules! start_of_octal_or_float_number {
    () => {
        [b'0', ..]
    };
}

macro_rules! start_of_float_number {
    () => {
        [b'.', ..]
    };
}

macro_rules! float_exponent {
    () => {
        [b'e' | b'E']
    };
}

macro_rules! float_separator {
    () => {
        [b'.', ..] | [b'e' | b'E', b'-' | b'+', b'0'..=b'9'] | [b'e' | b'E', b'0'..=b'9', ..]
    };
}

macro_rules! number_sign {
    () => {
        [b'-' | b'+']
    };
}

macro_rules! number_separator {
    () => {
        b'_'
    };
}

pub(crate) use float_exponent;
pub(crate) use float_separator;
pub(crate) use number_separator;
pub(crate) use number_sign;
pub(crate) use part_of_identifier;
pub(crate) use start_of_binary_number;
pub(crate) use start_of_float_number;
pub(crate) use start_of_hexadecimal_number;
pub(crate) use start_of_identifier;
pub(crate) use start_of_number;
pub(crate) use start_of_octal_number;
pub(crate) use start_of_octal_or_float_number;
