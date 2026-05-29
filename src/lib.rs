//! Forge Image — Decompose images into tiles for Plato agents

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageTile {
    pub id: Uuid,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub avg_brightness: f64,
    pub variance: f64,
    pub edge_density: f64,
    pub meta: HashMap<String, String>,
    pub index: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub total_pixels: u64,
    pub avg_brightness: f64,
    pub contrast: f64,
}

pub struct ImageDecomposer { pub channels: u8 }

impl ImageDecomposer {
    pub fn new(channels: u8) -> Self { Self { channels } }

    pub fn info(&self, pixels: &[u8], width: u32, height: u32) -> ImageInfo {
        let total_pixels = (width as u64) * (height as u64);
        let br = self.full_brightness(pixels, width, height);
        let var = self.full_variance(pixels, width, height, br);
        ImageInfo { width, height, channels: self.channels, total_pixels, avg_brightness: br, contrast: var.sqrt() }
    }

    fn pixel_brightness(&self, pixels: &[u8], idx: usize) -> f64 {
        let ch = self.channels as usize;
        if idx + ch > pixels.len() { return 0.0; }
        let sum: f64 = (0..ch).map(|c| pixels[idx + c] as f64).sum();
        sum / ch as f64
    }

    fn full_brightness(&self, pixels: &[u8], w: u32, h: u32) -> f64 {
        let ch = self.channels as usize;
        let stride = w as usize * ch;
        let n = (w as usize) * (h as usize) ;
        if n == 0 { return 0.0; }
        let mut sum = 0.0;
        for y in 0..h as usize {
            for x in 0..w as usize {
                sum += self.pixel_brightness(pixels, y * stride + x * ch);
            }
        }
        sum / n as f64
    }

    fn full_variance(&self, pixels: &[u8], w: u32, h: u32, mean: f64) -> f64 {
        let ch = self.channels as usize;
        let stride = w as usize * ch;
        let n = (w as usize) * (h as usize);
        if n == 0 { return 0.0; }
        let mut sum = 0.0;
        for y in 0..h as usize {
            for x in 0..w as usize {
                let b = self.pixel_brightness(pixels, y * stride + x * ch);
                sum += (b - mean).powi(2);
            }
        }
        sum / n as f64
    }

    fn region_brightness(&self, pixels: &[u8], w: u32, x0: u32, y0: u32, rw: u32, rh: u32) -> f64 {
        let ch = self.channels as usize;
        let stride = w as usize * ch;
        let n = (rw as usize) * (rh as usize);
        if n == 0 { return 0.0; }
        let mut sum = 0.0;
        for dy in 0..rh as usize {
            for dx in 0..rw as usize {
                let px = (y0 as usize + dy) * stride + (x0 as usize + dx) * ch;
                sum += self.pixel_brightness(pixels, px);
            }
        }
        sum / n as f64
    }

    fn region_variance(&self, pixels: &[u8], w: u32, x0: u32, y0: u32, rw: u32, rh: u32, mean: f64) -> f64 {
        let ch = self.channels as usize;
        let stride = w as usize * ch;
        let n = (rw as usize) * (rh as usize);
        if n == 0 { return 0.0; }
        let mut sum = 0.0;
        for dy in 0..rh as usize {
            for dx in 0..rw as usize {
                let px = (y0 as usize + dy) * stride + (x0 as usize + dx) * ch;
                let b = self.pixel_brightness(pixels, px);
                sum += (b - mean).powi(2);
            }
        }
        sum / n as f64
    }

    fn region_edge_density(&self, pixels: &[u8], w: u32, x0: u32, y0: u32, rw: u32, rh: u32) -> f64 {
        let ch = self.channels as usize;
        let stride = w as usize * ch;
        let mut edges = 0usize;
        let mut total = 0usize;
        for dy in 0..(rh as usize).saturating_sub(1) {
            for dx in 0..(rw as usize).saturating_sub(1) {
                let cx = x0 as usize + dx;
                let cy = y0 as usize + dy;
                let idx = cy * stride + cx * ch;
                let right = cy * stride + (cx + 1) * ch;
                let below = (cy + 1) * stride + cx * ch;
                let gx = (self.pixel_brightness(pixels, right) - self.pixel_brightness(pixels, idx)).abs();
                let gy = (self.pixel_brightness(pixels, below) - self.pixel_brightness(pixels, idx)).abs();
                if gx + gy > 30.0 { edges += 1; }
                total += 1;
            }
        }
        if total == 0 { 0.0 } else { edges as f64 / total as f64 }
    }

    pub fn grid_decompose(&self, pixels: &[u8], width: u32, height: u32, tile_size: u32) -> Vec<ImageTile> {
        let mut tiles = Vec::new();
        let mut idx = 0u64;
        let mut y = 0u32;
        while y < height {
            let mut x = 0u32;
            while x < width {
                let tw = tile_size.min(width - x);
                let th = tile_size.min(height - y);
                let br = self.region_brightness(pixels, width, x, y, tw, th);
                let var = self.region_variance(pixels, width, x, y, tw, th, br);
                let edge = self.region_edge_density(pixels, width, x, y, tw, th);
                let mut meta = HashMap::new();
                meta.insert("grid_x".into(), (x / tile_size).to_string());
                meta.insert("grid_y".into(), (y / tile_size).to_string());
                tiles.push(ImageTile { id: Uuid::new_v4(), x, y, width: tw, height: th, avg_brightness: br, variance: var, edge_density: edge, meta, index: idx as u64 });
                idx += 1;
                x += tile_size;
            }
            y += tile_size;
        }
        tiles
    }

    pub fn strip_decompose(&self, pixels: &[u8], width: u32, height: u32, strip_height: u32) -> Vec<ImageTile> {
        let mut tiles = Vec::new();
        let mut y = 0u32;
        let mut idx = 0u64;
        while y < height {
            let sh = strip_height.min(height - y);
            let br = self.region_brightness(pixels, width, 0, y, width, sh);
            let var = self.region_variance(pixels, width, 0, y, width, sh, br);
            let edge = self.region_edge_density(pixels, width, 0, y, width, sh);
            tiles.push(ImageTile { id: Uuid::new_v4(), x: 0, y, width, height: sh, avg_brightness: br, variance: var, edge_density: edge, meta: HashMap::new(), index: idx });
            idx += 1;
            y += strip_height;
        }
        tiles
    }

    pub fn column_decompose(&self, pixels: &[u8], width: u32, height: u32, col_width: u32) -> Vec<ImageTile> {
        let mut tiles = Vec::new();
        let mut x = 0u32;
        let mut idx = 0u64;
        while x < width {
            let cw = col_width.min(width - x);
            let br = self.region_brightness(pixels, width, x, 0, cw, height);
            let var = self.region_variance(pixels, width, x, 0, cw, height, br);
            let edge = self.region_edge_density(pixels, width, x, 0, cw, height);
            tiles.push(ImageTile { id: Uuid::new_v4(), x, y: 0, width: cw, height, avg_brightness: br, variance: var, edge_density: edge, meta: HashMap::new(), index: idx });
            idx += 1;
            x += col_width;
        }
        tiles
    }

    pub fn conservation_ratio(&self, original: &[u8], tiles: &[ImageTile], width: u32, height: u32) -> f64 {
        let total = (width as u64) * (height as u64);
        let covered: u64 = tiles.iter().map(|t| (t.width as u64) * (t.height as u64)).sum();
        if total == 0 { return 1.0; }
        (covered as f64 / total as f64).min(1.0)
    }

    pub fn reconstruct_validate(&self, tiles: &[ImageTile], width: u32, height: u32) -> bool {
        let total = (width as u64) * (height as u64);
        let covered: u64 = tiles.iter().map(|t| (t.width as u64) * (t.height as u64)).sum();
        covered == total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_image(w: u32, h: u32) -> Vec<u8> { vec![255u8; (w * h * 4) as usize] }
    fn black_image(w: u32, h: u32) -> Vec<u8> { vec![0u8; (w * h * 4) as usize] }
    fn checker_image(w: u32, h: u32) -> Vec<u8> {
        let mut p = Vec::new();
        for y in 0..h { for x in 0..w {
            let v = if (x + y) % 2 == 0 { 255 } else { 0 };
            p.extend_from_slice(&[v, v, v, 255]);
        }}
        p
    }

    #[test] fn test_info_white() {
        let d = ImageDecomposer::new(4);
        let info = d.info(&white_image(10, 10), 10, 10);
        assert_eq!(info.width, 10); assert_eq!(info.height, 10);
        assert!((info.avg_brightness - 255.0).abs() < 1.0);
    }

    #[test] fn test_info_black() {
        let d = ImageDecomposer::new(4);
        let info = d.info(&black_image(10, 10), 10, 10);
        assert!(info.avg_brightness.abs() < 1.0);
        assert!(info.contrast.abs() < 1.0);
    }

    #[test] fn test_grid_exact() {
        let d = ImageDecomposer::new(4);
        let tiles = d.grid_decompose(&white_image(20, 20), 20, 20, 10);
        assert_eq!(tiles.len(), 4);
    }

    #[test] fn test_grid_nonexact() {
        let d = ImageDecomposer::new(4);
        let tiles = d.grid_decompose(&white_image(15, 15), 15, 15, 10);
        assert_eq!(tiles.len(), 4);
        // bottom-right tile is 5x5
        let last = tiles.last().unwrap();
        assert_eq!(last.width, 5); assert_eq!(last.height, 5);
    }

    #[test] fn test_strip_decompose() {
        let d = ImageDecomposer::new(4);
        let tiles = d.strip_decompose(&white_image(20, 30), 20, 30, 10);
        assert_eq!(tiles.len(), 3);
        assert_eq!(tiles[0].width, 20);
    }

    #[test] fn test_column_decompose() {
        let d = ImageDecomposer::new(4);
        let tiles = d.column_decompose(&white_image(30, 20), 30, 20, 10);
        assert_eq!(tiles.len(), 3);
        assert_eq!(tiles[0].height, 20);
    }

    #[test] fn test_brightness_checker() {
        let d = ImageDecomposer::new(4);
        let br = d.region_brightness(&checker_image(4, 4), 4, 0, 0, 4, 4);
        assert!((br - 159.375).abs() < 5.0);
    }

    #[test] fn test_variance_checker() {
        let d = ImageDecomposer::new(4);
        let mean = d.region_brightness(&checker_image(4, 4), 4, 0, 0, 4, 4);
        let var = d.region_variance(&checker_image(4, 4), 4, 0, 0, 4, 4, mean);
        assert!(var > 1000.0); // high contrast
    }

    #[test] fn test_edge_density_checker() {
        let d = ImageDecomposer::new(4);
        let edge = d.region_edge_density(&checker_image(10, 10), 10, 0, 0, 10, 10);
        assert!(edge > 0.5); // checkerboard has many edges
    }

    #[test] fn test_conservation_perfect() {
        let d = ImageDecomposer::new(4);
        let tiles = d.grid_decompose(&white_image(20, 20), 20, 20, 10);
        let cr = d.conservation_ratio(&white_image(20, 20), &tiles, 20, 20);
        assert!((cr - 1.0).abs() < 0.001);
    }

    #[test] fn test_reconstruct_validate() {
        let d = ImageDecomposer::new(4);
        let tiles = d.grid_decompose(&white_image(20, 20), 20, 20, 10);
        assert!(d.reconstruct_validate(&tiles, 20, 20));
    }

    #[test] fn test_1x1_image() {
        let d = ImageDecomposer::new(4);
        let tiles = d.grid_decompose(&white_image(1, 1), 1, 1, 10);
        assert_eq!(tiles.len(), 1);
    }

    #[test] fn test_tile_serialization() {
        let d = ImageDecomposer::new(4);
        let tiles = d.grid_decompose(&white_image(10, 10), 10, 10, 5);
        let json = serde_json::to_string(&tiles).unwrap();
        let back: Vec<ImageTile> = serde_json::from_str(&json).unwrap();
        assert_eq!(tiles.len(), back.len());
    }

    #[test] fn test_info_total_pixels() {
        let d = ImageDecomposer::new(4);
        let info = d.info(&white_image(100, 50), 100, 50);
        assert_eq!(info.total_pixels, 5000);
    }
}
