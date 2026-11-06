use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Redirect, Response},
};
use tower::ServiceExt;
use tower_http::services::{ServeDir, ServeFile};

use crate::{auth::current, util::{error, wants_html}, Shared};

fn no_store(mut res: Response) -> Response {
    res.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    res
}

fn safe_next(path: &str) -> String {
    if path.starts_with('/') && !path.starts_with("//") && !path.contains('\\') && path.len() < 200 {
        path.to_string()
    } else {
        "/".to_string()
    }
}

pub async fn login_page(State(state): State<Shared>, req: Request) -> Response {
    if let Some(s) = current(&state, req.headers()).await {
        if !s.restricted {
            return Redirect::to("/").into_response();
        }
    }
    let file = state.web_public.join("login.html");
    match ServeFile::new(file).oneshot(req).await {
        Ok(res) => no_store(res.map(Body::new)),
        Err(_) => error(StatusCode::INTERNAL_SERVER_ERROR, "Login page unavailable."),
    }
}

pub async fn public_files(State(state): State<Shared>, req: Request) -> Response {
    let path = req.uri().path().trim_start_matches("/public").to_string();
    let (mut parts, body) = req.into_parts();
    parts.uri = match path.parse() {
        Ok(u) => u,
        Err(_) => return error(StatusCode::BAD_REQUEST, "Invalid path."),
    };
    let req = Request::from_parts(parts, body);
    match ServeDir::new(&state.web_public).oneshot(req).await {
        Ok(mut res) => {
            res.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            res.map(Body::new)
        }
        Err(_) => error(StatusCode::NOT_FOUND, "Not found."),
    }
}

pub async fn app_files(State(state): State<Shared>, req: Request) -> Response {
    let path = req.uri().path().to_string();
    if path.contains("..") || path.contains('\\') {
        return error(StatusCode::BAD_REQUEST, "Invalid path.");
    }
    match current(&state, req.headers()).await {
        Some(s) if !s.restricted => match ServeDir::new(&state.web_app).append_index_html_on_directories(true).oneshot(req).await {
            Ok(res) => no_store(res.map(Body::new)),
            Err(_) => error(StatusCode::NOT_FOUND, "Not found."),
        },
        Some(_) => Redirect::to("/login?change=1").into_response(),
        None if wants_html(req.headers()) || path == "/" => {
            let next = safe_next(&path);
            let target = if next == "/" { "/login".to_string() } else { format!("/login?next={}", urlencode(&next)) };
            no_store(Redirect::to(&target).into_response())
        }
        None => error(StatusCode::UNAUTHORIZED, "Sign in to continue."),
    }
}

fn urlencode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_paths_stay_on_site() {
        assert_eq!(safe_next("/investigate"), "/investigate");
        assert_eq!(safe_next("//evil.example"), "/");
        assert_eq!(safe_next("https://evil.example"), "/");
        assert_eq!(urlencode("/a b?c"), "/a%20b%3Fc");
    }
}
