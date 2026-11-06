use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use steamid_ng::SteamID;

#[derive(Debug)]
pub enum Parameter {
    Float(f32),
    Int(i32),
    Bool(bool),
    UInt(u64),
    String(String),
}

#[derive(Debug)]
pub enum ParameterError {
    TypeMismatch,
}

impl TryFrom<&Parameter> for f32 {
    type Error = ParameterError;

    fn try_from(param: &Parameter) -> Result<Self, Self::Error> {
        match param {
            Parameter::Float(f) => Ok(*f),
            Parameter::Int(i) => Ok(*i as f32),
            Parameter::UInt(u) => Ok(*u as f32),
            _ => Err(ParameterError::TypeMismatch),
        }
    }
}

impl TryFrom<&Parameter> for i32 {
    type Error = ParameterError;

    fn try_from(param: &Parameter) -> Result<Self, Self::Error> {
        match param {
            Parameter::Int(i) => Ok(*i),
            Parameter::UInt(u) if *u <= i32::MAX as u64 => Ok(*u as i32),
            Parameter::Float(f) if f.fract() == 0.0 => Ok(*f as i32),
            _ => Err(ParameterError::TypeMismatch),
        }
    }
}

impl TryFrom<&Parameter> for u64 {
    type Error = ParameterError;

    fn try_from(param: &Parameter) -> Result<Self, Self::Error> {
        match param {
            Parameter::UInt(u) => Ok(*u),
            Parameter::Int(i) if *i >= 0 => Ok(*i as u64),
            Parameter::Float(f) if *f >= 0.0 && f.fract() == 0.0 => Ok(*f as u64),
            Parameter::String(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() || trimmed == "0" {
                    return Ok(0);
                }
                if let Ok(val) = trimmed.parse::<u64>() {
                    return Ok(val);
                }
                if let Ok(steam) = SteamID::from_steam3(trimmed) {
                    return Ok(steam.into());
                }
                if let Ok(steam) = SteamID::from_steam2(trimmed) {
                    return Ok(steam.into());
                }
                Err(ParameterError::TypeMismatch)
            }
            _ => Err(ParameterError::TypeMismatch),
        }
    }
}

impl TryFrom<&Parameter> for String {
    type Error = ParameterError;

    fn try_from(param: &Parameter) -> Result<Self, Self::Error> {
        match param {
            Parameter::String(s) => Ok(s.clone()),
            Parameter::UInt(u) => Ok(u.to_string()),
            Parameter::Int(i) => Ok(i.to_string()),
            Parameter::Float(f) => Ok(f.to_string()),
            Parameter::Bool(b) => Ok(b.to_string()),
        }
    }
}

impl TryFrom<&Parameter> for bool {
    type Error = ParameterError;

    fn try_from(param: &Parameter) -> Result<Self, Self::Error> {
        if let Parameter::Bool(b) = param {
            Ok(*b)
        } else {
            Err(ParameterError::TypeMismatch)
        }
    }
}

impl TryFrom<&Parameter> for Parameter {
    type Error = ParameterError;

    fn try_from(param: &Parameter) -> Result<Self, Self::Error> {
        Ok(param.clone())
    }
}

impl Clone for Parameter {
    fn clone(&self) -> Self {
        match self {
            Parameter::Float(f) => Parameter::Float(*f),
            Parameter::Int(i) => Parameter::Int(*i),
            Parameter::Bool(b) => Parameter::Bool(*b),
            Parameter::UInt(u) => Parameter::UInt(*u),
            Parameter::String(s) => Parameter::String(s.clone()),
        }
    }
}

impl Serialize for Parameter {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Parameter::Float(f) => serializer.serialize_f32(*f),
            Parameter::Int(i) => serializer.serialize_i32(*i),
            Parameter::Bool(b) => serializer.serialize_bool(*b),
            Parameter::UInt(u) => serializer.serialize_u64(*u),
            Parameter::String(s) => serializer.serialize_str(s),
        }
    }
}

impl<'a> Deserialize<'a> for Parameter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'a>,
    {
        struct ParameterVisitor;

        impl<'de> serde::de::Visitor<'de> for ParameterVisitor {
            type Value = Parameter;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an f32, i32, u64, string, or bool")
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Parameter::Float(value as f32))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value >= i32::MIN as i64 && value <= i32::MAX as i64 {
                    Ok(Parameter::Int(value as i32))
                } else if value >= 0 {
                    Ok(Parameter::UInt(value as u64))
                } else {
                    Ok(Parameter::Int(value as i32))
                }
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                if value <= i32::MAX as u64 {
                    Ok(Parameter::Int(value as i32))
                } else {
                    Ok(Parameter::UInt(value))
                }
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Parameter::String(value.to_string()))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Parameter::String(value))
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(Parameter::Bool(value))
            }
        }

        deserializer.deserialize_any(ParameterVisitor)
    }
}

// Maps parameter names to their values.
pub type Parameters = HashMap<String, Parameter>;

/// Parameters that only apply to the running session: they are never saved to or restored
/// from a params file, so every launch starts from the algorithm default (e.g. the recorder
/// SteamID override falls back to auto-detection, a target player to all players).
pub const SESSION_ONLY_PARAMETERS: &[&str] = &["recorder_id", "target_player"];

/// Removes every session-only parameter from a full config map.
pub fn strip_session_only_parameters(config: &mut HashMap<String, Parameters>) {
    for params in config.values_mut() {
        params.retain(|key, _| !SESSION_ONLY_PARAMETERS.contains(&key.as_str()));
    }
}

// Maps algorithm names to their parameters.
pub type Config = HashMap<String, Parameters>;

pub fn get_parameter_value<T>(params: &Parameters, param_name: &str) -> T
where
    T: for<'a> TryFrom<&'a Parameter, Error = ParameterError>,
{
    match params.get(param_name) {
        Some(param) => match T::try_from(param) {
            Ok(value) => value,
            Err(_) => panic!("Parameter {} has wrong type", param_name),
        },
        None => panic!("Parameter {} not found", param_name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64_from_parameters() {
        let sid64: u64 = 76561198012345678;

        // UInt
        let p_uint = Parameter::UInt(sid64);
        assert_eq!(u64::try_from(&p_uint).unwrap(), sid64);

        // Int
        let p_int = Parameter::Int(123456);
        assert_eq!(u64::try_from(&p_int).unwrap(), 123456);

        // String full SteamID64
        let p_str_sid = Parameter::String("76561198012345678".to_string());
        assert_eq!(u64::try_from(&p_str_sid).unwrap(), sid64);

        // String SteamID3
        let p_str_steam3 = Parameter::String("[U:1:52079950]".to_string());
        assert_eq!(u64::try_from(&p_str_steam3).unwrap(), 76561198012345678);

        // String SteamID2
        let p_str_steam2 = Parameter::String("STEAM_0:0:26039975".to_string());
        assert_eq!(u64::try_from(&p_str_steam2).unwrap(), 76561198012345678);

        // Empty / zero string
        let p_empty = Parameter::String("".to_string());
        assert_eq!(u64::try_from(&p_empty).unwrap(), 0);

        let p_zero = Parameter::String("0".to_string());
        assert_eq!(u64::try_from(&p_zero).unwrap(), 0);
    }
}