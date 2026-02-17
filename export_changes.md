# Export Format Changes: v1.1 → v2.0

Breaking changes to the windowsill binary export format driven by hierarchy
growth: 5 archipelagos (was 4), 67 islands (was 52), 283 reefs (was 233).
All three exceeded their old bit-width limits.

## Summary of Changes

| What | Old (v1.1) | New (v2.0) |
|------|-----------|------------|
| `FORMAT_VERSION` | `"1.1"` | `"2.0"` |
| `hierarchy_addr` | u16: `arch(2)\|island(6)\|reef(8)` | u32: `arch(8)\|island(8)\|reef(16)` |
| Reef ID width | u8 (max 255) | u16 (max 65535) |
| Island ID width | 6-bit (max 63) | u8 (max 255) |
| Arch ID width | 2-bit (max 3) | u8 (max 255) |

## Hierarchy Counts

| Level | Old | New |
|-------|-----|-----|
| Archipelagos | 4 | 5 |
| Islands | 52 | 67 |
| Reefs | 233 | 283 |

---

## V2 (flat binary) File Changes

### `reef_edges.bin` (WSRE)

Record stride changed from 6 → 8 bytes.

| Field | Old | New |
|-------|-----|-----|
| `src_reef_id` | u8 | **u16** |
| `tgt_reef_id` | u8 | **u16** |
| `weight` | f32 | f32 |
| **Record size** | **6 bytes** | **8 bytes** |

**Rust struct change:**
```rust
// Old
#[repr(C, packed)]
struct ReefEdge {
    src: u8,
    tgt: u8,
    weight: f32,
}

// New
#[repr(C, packed)]
struct ReefEdge {
    src: u16,
    tgt: u16,
    weight: f32,
}
```

### `word_reefs.bin` (WSWR)

Data record layout changed (same 4-byte stride).

| Field | Old | New |
|-------|-----|-----|
| `reef_id` | u8 + 1 pad byte | **u16** |
| `bm25_q` | u16 | u16 |
| **Record size** | **4 bytes** | **4 bytes** |

The index section (offset/count pairs, u32 each) is unchanged.

**Rust struct change:**
```rust
// Old
#[repr(C, packed)]
struct WordReefEntry {
    reef_id: u8,
    _pad: u8,
    bm25_q: u16,
}

// New
#[repr(C, packed)]
struct WordReefEntry {
    reef_id: u16,
    bm25_q: u16,
}
```

### `reef_meta.bin` (WSRM)

Record stride changed from 68 → 72 bytes.

| Field | Old | New |
|-------|-----|-----|
| `hierarchy_addr` | u16 | **u32** |
| `n_words` | u16 | **u32** |
| `name` | 64 bytes (null-padded UTF-8) | 64 bytes (null-padded UTF-8) |
| **Record size** | **68 bytes** | **72 bytes** |

**Rust struct change:**
```rust
// Old
#[repr(C, packed)]
struct ReefMeta {
    hierarchy_addr: u16,
    n_words: u16,
    name: [u8; 64],
}

// New
#[repr(C, packed)]
struct ReefMeta {
    hierarchy_addr: u32,
    n_words: u32,
    name: [u8; 64],
}
```

### `hierarchy_addr` Unpacking

```rust
// Old (u16): arch(2)|island(6)|reef(8)
let arch   = (addr >> 14) & 0x03;
let island = (addr >> 8)  & 0x3F;
let reef   =  addr        & 0xFF;

// New (u32): arch(8)|island(8)|reef(16)
let arch   = (addr >> 24) & 0xFF;
let island = (addr >> 16) & 0xFF;
let reef   =  addr        & 0xFFFF;
```

### Unchanged Files

These v2 files have **no format changes**:

- `word_lookup.bin` (WSWL) — 24-byte records, no reef IDs
- `island_meta.bin` (WSIM) — `arch_id` remains u8 (5 < 255), 66-byte records
- `background.bin` (WSBG) — `f32[N_REEFS]` arrays, N_REEFS is in the header
- `compounds.bin` (WSCP) — index + string pool, no reef IDs
- `constants.bin` (WSCN) — scalars already u32/f32, array lengths from header

Note: `background.bin` and `constants.bin` arrays are sized by `N_REEFS` from
their headers, so they grow from 233 to 283 entries automatically. No struct
changes needed — just ensure the consumer reads the count from the header rather
than hardcoding 233.

---

## V1 (msgpack) Changes

The v1 msgpack format uses dynamic integer encoding, so no field width changes
are needed in the deserialization code. The only semantic change is:

- `hierarchy_addr` values in `reef_meta.bin` are now u32-range integers packed
  with the new `arch(8)|island(8)|reef(16)` layout. Any code that unpacks the
  bit fields needs the updated shift/mask constants shown above.

---

## New Data Characteristics

| Metric | Old | New |
|--------|-----|-----|
| avg_reef_words | ~5,733 | ~4,731 |
| Background model arrays | 233 entries | 283 entries |
| Reef edges | ~3,700 | ~7,555 |
| BM25 scores | u16 (unchanged) | u16 (unchanged) |

The new 5th archipelago ("specialized activities and practices") contains
domain-specific reefs created from artificial dimensions. These reefs tend to
be smaller (fewer words) than natural reefs, which lowers `avg_reef_words`.

---

## Migration Checklist

1. Update `hierarchy_addr` unpacking: u16 → u32, new bit layout
2. Update `reef_id` types: u8 → u16 everywhere
3. Update `reef_edges.bin` record parsing: 6 → 8 byte stride
4. Update `reef_meta.bin` record parsing: 68 → 72 byte stride
5. Update `word_reefs.bin` data record parsing: `u8+pad+u16` → `u16+u16`
6. Remove any hardcoded `N_REEFS=233`, `N_ISLANDS=52`, `N_ARCHS=4`
7. Verify background/constants array sizes come from file headers
