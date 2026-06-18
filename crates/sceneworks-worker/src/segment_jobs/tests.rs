use super::*;
use serde_json::json;

fn obj(value: Value) -> JsonObject {
    value.as_object().unwrap().clone()
}

#[test]
fn parse_box_accepts_xyxy_array() {
    let payload = obj(json!({ "box": [10.0, 20.0, 110.0, 220.0] }));
    assert_eq!(parse_box(&payload).unwrap(), [10.0, 20.0, 110.0, 220.0]);
}

#[test]
fn parse_box_accepts_rect_object() {
    // {x,y,width,height} → [x, y, x+w, y+h] (the editor's rect shape).
    let payload = obj(json!({ "box": { "x": 10.0, "y": 20.0, "width": 100.0, "height": 200.0 } }));
    assert_eq!(parse_box(&payload).unwrap(), [10.0, 20.0, 110.0, 220.0]);
}

#[test]
fn parse_box_accepts_integer_json_numbers() {
    let payload = obj(json!({ "box": [10, 20, 110, 220] }));
    assert_eq!(parse_box(&payload).unwrap(), [10.0, 20.0, 110.0, 220.0]);
}

#[test]
fn parse_box_rejects_missing_or_malformed() {
    assert!(parse_box(&obj(json!({}))).is_err());
    assert!(parse_box(&obj(json!({ "box": [1.0, 2.0, 3.0] }))).is_err());
    assert!(parse_box(&obj(json!({ "box": "nope" }))).is_err());
    assert!(parse_box(&obj(json!({ "box": { "x": 1.0, "y": 2.0 } }))).is_err());
}
