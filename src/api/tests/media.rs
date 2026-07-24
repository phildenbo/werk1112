use super::support::*;

#[test]
fn media_request_shapes_normalize_openai_and_werk_fields() {
    let parsed: ImageGenerationApiRequest = serde_json::from_value(json!({
        "model": "media",
        "prompt": "a small station",
        "negative_prompt": "crowded",
        "n": 2,
        "size": "640x480",
        "response_format": "b64_json",
        "steps": 12,
        "parameters": {"guidance": 4.5},
        "backend": "mock-media",
        "quality": "hd",
        "allow_cpu_offload": true
    }))
    .unwrap();
    let (request, response_format) = parsed.into_inference().unwrap();
    assert_eq!(request.task, InferenceTask::ImageGeneration);
    assert!(matches!(response_format, DirectResponseFormat::Base64));
    assert_eq!(
        request
            .parameters
            .get("image.width")
            .and_then(ParameterValue::as_u64),
        Some(640)
    );
    assert_eq!(
        request
            .parameters
            .get("image.height")
            .and_then(ParameterValue::as_u64),
        Some(480)
    );
    assert_eq!(
        request
            .parameters
            .get("image.num_images")
            .and_then(ParameterValue::as_u64),
        Some(2)
    );
    assert_eq!(
        request
            .parameters
            .get("image.steps")
            .and_then(ParameterValue::as_u64),
        Some(12)
    );
    assert_eq!(
        request
            .parameters
            .get("image.guidance")
            .and_then(ParameterValue::as_f64),
        Some(4.5)
    );
    assert_eq!(request.routing.backend.as_deref(), Some("mock-media"));
    assert_eq!(request.routing.quality.as_deref(), Some("high"));
    assert_eq!(request.routing.allow_cpu_offload, OverrideBool::Enabled);

    let parsed: ImageEditApiRequest = serde_json::from_value(json!({
        "model": "media",
        "prompt": "replace the sky",
        "image": "data:image/png;base64,AAEC"
    }))
    .unwrap();
    let (request, _) = parsed.into_inference().unwrap();
    assert!(matches!(
        request.inputs[0].source,
        InferenceInputSource::Base64 { ref data } if data == "AAEC"
    ));
    assert_eq!(request.inputs[0].mime_type.as_deref(), Some("image/png"));

    let serialized = serde_json::to_value(ApiMediaInput::Object(ApiMediaInputObject {
        path: None,
        url: Some("https://example.test/input.wav".to_string()),
        base64: None,
        mime_type: Some("audio/wav".to_string()),
    }))
    .unwrap();
    assert_eq!(
        serialized["url"],
        Value::String("https://example.test/input.wav".to_string())
    );
}

#[tokio::test]
async fn direct_media_routes_return_openai_data_and_werk_metadata() {
    let app = media_app(Vec::new());

    let response = post_json(
        &app,
        "/v1/images/generations",
        json!({
            "model": "media",
            "prompt": "an orbital greenhouse",
            "size": "512x512",
            "response_format": "b64_json"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["werk"]["task"], "image_generation");
    assert_eq!(value["data"][0]["mime_type"], "image/png");
    assert_eq!(value["data"][0]["b64_json"], encode_base64(b"mock image"));
    assert_eq!(
        value["werk"]["effective_request"]["parameters"]["image.width"]["value"],
        512
    );

    let response = post_json(
        &app,
        "/v1/images/edits",
        json!({
            "model": "media",
            "prompt": "make it blue",
            "image": {"base64": "AAEC", "mime_type": "image/png"}
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["werk"]["task"], "image_editing");
    let output_url = value["data"][0]["url"].as_str().unwrap().to_string();
    assert!(output_url.starts_with("/v1/outputs/"));
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(output_url)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "image/png"
    );
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(&bytes[..], b"mock image");

    let response = post_json(
        &app,
        "/v1/audio/speech",
        json!({
            "model": "media",
            "input": "hello",
            "voice": "test",
            "response_format": "wav"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "audio/wav"
    );
    assert!(
        response
            .headers()
            .get("x-werk-output-id")
            .is_some_and(|value| value.to_str().unwrap().starts_with("out-"))
    );
    let bytes = body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(&bytes[..], b"mock audio");

    let response = post_json(
        &app,
        "/v1/audio/transcriptions",
        json!({
            "model": "media",
            "file": {"base64": "AAEC", "mime_type": "audio/wav"},
            "response_format": "text"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let value = response_json(response).await;
    assert_eq!(value["werk"]["task"], "speech_to_text");
    assert_eq!(value["data"][0]["text"], "mock transcript");
}

#[tokio::test]
async fn long_media_and_generic_job_routes_return_job_records() {
    let app = media_app(Vec::new());

    let response = post_json(
        &app,
        "/v1/videos/generations",
        json!({
            "model": "media",
            "prompt": "slow camera orbit",
            "size": "832x480",
            "response_format": "mp4"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let video_job = response_json(response).await;
    assert_eq!(video_job["request"]["task"], "video_generation");
    let video_id = video_job["id"].as_str().unwrap().to_string();

    let response = post_json(
        &app,
        "/v1/audio/generations",
        json!({
            "model": "media",
            "prompt": "quiet analogue ambient",
            "task": "music-generation",
            "n": 2,
            "response_format": "wav"
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let audio_job = response_json(response).await;
    assert_eq!(audio_job["request"]["task"], "music_generation");

    let response = post_json(
        &app,
        "/v1/audio/speech",
        json!({
            "model": "media",
            "input": "background speech",
            "async": true
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);

    let response = post_json(
        &app,
        "/v1/jobs",
        json!({
            "model": "media",
            "task": "image-generation",
            "prompt": "generic job request",
            "parameters": {"width": 512, "height": 512}
        }),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::ACCEPTED);
    let generic_job = response_json(response).await;
    assert_eq!(generic_job["request"]["task"], "image_generation");

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/jobs/{video_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let fetched = response_json(response).await;
    assert_eq!(fetched["id"], video_id);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/jobs/{video_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let cancelled = response_json(response).await;
    assert!(matches!(
        cancelled["status"].as_str(),
        Some("cancelled" | "completed" | "failed")
    ));
}

#[tokio::test]
async fn capability_and_parameter_routes_are_authenticated() {
    let app = media_app(vec!["sk-media".to_string()]);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = post_json(
        &app,
        "/v1/images/generations",
        json!({"model": "media", "prompt": "unauthorized"}),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/capabilities")
                .header(header::AUTHORIZATION, "Bearer sk-media")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let capabilities = response_json(response).await;
    assert_eq!(capabilities["object"], "werk.capabilities");
    assert!(
        capabilities["models"][0]["available_tasks"]
            .as_array()
            .is_some_and(|tasks| tasks.contains(&json!("image_generation")))
    );

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/parameters?task=image-generation&model=media&backend=mock-media")
                .header(header::AUTHORIZATION, "Bearer sk-media")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let parameters = response_json(response).await;
    assert_eq!(parameters["object"], "werk.parameter_schema");
    assert_eq!(parameters["task"], "image_generation");
    assert_eq!(parameters["parameter_support"]["image.width"], "native");
    assert_eq!(parameters["runtime_candidates"][0]["id"], "mock-media-cpu");
    assert!(
        parameters["parameters"]
            .as_array()
            .unwrap()
            .iter()
            .any(|parameter| parameter["path"] == "image.width")
    );

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/parameters?task=image-generation&backend=cuda")
                .header(header::AUTHORIZATION, "Bearer sk-media")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let parameters = response_json(response).await;
    assert_eq!(parameters["backend"], "cuda");
    assert!(
        parameters["runtime_candidates"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}
