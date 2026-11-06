use serde_json::Value;

type Ring = Vec<(f64, f64)>;

pub struct Geofence {
    polygons: Vec<Vec<Ring>>,
    pub name: String,
}

fn ring(value: &Value) -> Option<Ring> {
    value.as_array()?.iter().map(|p| Some((p.get(0)?.as_f64()?, p.get(1)?.as_f64()?))).collect()
}

fn polygon(value: &Value) -> Option<Vec<Ring>> {
    value.as_array()?.iter().map(ring).collect()
}

fn geometries(value: &Value, out: &mut Vec<Vec<Ring>>) {
    match value.get("type").and_then(Value::as_str) {
        Some("FeatureCollection") => {
            for f in value.get("features").and_then(Value::as_array).into_iter().flatten() {
                geometries(f, out);
            }
        }
        Some("Feature") => {
            if let Some(g) = value.get("geometry") {
                geometries(g, out);
            }
        }
        Some("Polygon") => {
            if let Some(p) = value.get("coordinates").and_then(polygon) {
                out.push(p);
            }
        }
        Some("MultiPolygon") => {
            for p in value.get("coordinates").and_then(Value::as_array).into_iter().flatten() {
                if let Some(p) = polygon(p) {
                    out.push(p);
                }
            }
        }
        _ => {}
    }
}

fn in_ring(ring: &Ring, x: f64, y: f64) -> bool {
    let mut inside = false;
    let n = ring.len();
    if n < 3 {
        return false;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = ring[i];
        let (xj, yj) = ring[j];
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

impl Geofence {
    pub fn load(path: &str) -> Result<Geofence, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let value: Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        let mut polygons = Vec::new();
        geometries(&value, &mut polygons);
        if polygons.is_empty() {
            return Err(format!("{path}: no polygon found"));
        }
        let name = value
            .pointer("/features/0/properties/name")
            .and_then(Value::as_str)
            .unwrap_or("Bengaluru")
            .to_string();
        Ok(Geofence { polygons, name })
    }

    pub fn contains(&self, lon: f64, lat: f64) -> bool {
        self.polygons.iter().any(|rings| {
            rings.first().map(|outer| in_ring(outer, lon, lat)).unwrap_or(false) && !rings.iter().skip(1).any(|hole| in_ring(hole, lon, lat))
        })
    }

    pub fn distance_m(&self, lon: f64, lat: f64) -> f64 {
        if self.contains(lon, lat) {
            return 0.0;
        }
        let kx = 111_320.0 * lat.to_radians().cos();
        let ky = 110_540.0;
        let mut best = f64::INFINITY;
        for rings in &self.polygons {
            for r in rings {
                for w in r.windows(2) {
                    let (ax, ay) = ((w[0].0 - lon) * kx, (w[0].1 - lat) * ky);
                    let (bx, by) = ((w[1].0 - lon) * kx, (w[1].1 - lat) * ky);
                    let (dx, dy) = (bx - ax, by - ay);
                    let len2 = dx * dx + dy * dy;
                    let t = if len2 > 0.0 { (-(ax * dx + ay * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
                    let (px, py) = (ax + t * dx, ay + t * dy);
                    best = best.min((px * px + py * py).sqrt());
                }
            }
        }
        best
    }

    pub fn admits(&self, lon: f64, lat: f64, margin_m: f64) -> bool {
        lon.is_finite() && lat.is_finite() && (self.contains(lon, lat) || self.distance_m(lon, lat) <= margin_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Geofence {
        let ring = vec![(77.0, 12.0), (78.0, 12.0), (78.0, 13.0), (77.0, 13.0), (77.0, 12.0)];
        let hole = vec![(77.4, 12.4), (77.6, 12.4), (77.6, 12.6), (77.4, 12.6), (77.4, 12.4)];
        Geofence { polygons: vec![vec![ring, hole]], name: "test".into() }
    }

    #[test]
    fn inside_outside_and_holes() {
        let g = square();
        assert!(g.contains(77.2, 12.2));
        assert!(!g.contains(77.5, 12.5));
        assert!(!g.contains(76.5, 12.5));
    }

    #[test]
    fn margin_admits_points_just_outside() {
        let g = square();
        assert!(g.distance_m(76.99, 12.5) < 1_200.0);
        assert!(g.admits(76.99, 12.5, 2_000.0));
        assert!(!g.admits(76.5, 12.5, 2_000.0));
        assert!(!g.admits(f64::NAN, 12.5, 2_000.0));
    }
}
