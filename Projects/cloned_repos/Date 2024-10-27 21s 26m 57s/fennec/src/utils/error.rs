use std::error::Error;

pub fn print(error: impl Error) {
    fennec_feedback::error!(target = "fennec", "{}", error);
    fennec_feedback::debug!(target = "fennec", "{:#?}", error);

    if let Some(source) = error.source() {
        fennec_feedback::debug!(target = "fennec", "{:#?}", source);
    }
}

pub fn bail<T>(error: impl Error) -> T {
    print(error);

    std::process::exit(1);
}
