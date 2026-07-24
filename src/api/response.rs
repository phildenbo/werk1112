use axum::{
    Json,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};

use crate::openai::{ErrorObject, ErrorResponse};

pub(super) fn api_error(status: StatusCode, message: String, param: Option<String>) -> Response {
    (
        status,
        Json(ErrorResponse {
            error: ErrorObject {
                message,
                kind: "invalid_request_error".to_string(),
                param,
                code: None,
            },
        }),
    )
        .into_response()
}

pub(super) fn auth_error(message: &'static str) -> Response {
    let mut response = api_error(StatusCode::UNAUTHORIZED, message.to_string(), None);
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
}

pub(super) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = a.len() ^ b.len();
    let max_len = a.len().max(b.len());
    for index in 0..max_len {
        let left = a.get(index).copied().unwrap_or(0);
        let right = b.get(index).copied().unwrap_or(0);
        diff |= (left ^ right) as usize;
    }
    diff == 0
}
