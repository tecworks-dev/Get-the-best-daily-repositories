use clap::ValueEnum;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum DevelopmentType {
    Backend,
    Frontend,
    Fullstack,
    DataScience,
    DevOps,
    Blockchain,
    MachineLearning,
    SystemsProgramming,
    GameDevelopment,
    Security,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum JargonLevel {
    Low,
    Medium,
    High,
    Extreme,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum Complexity {
    Low,
    Medium,
    High,
    Extreme,
}
