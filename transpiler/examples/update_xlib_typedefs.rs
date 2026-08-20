//! Regenerates `src/parser/xlib_typedefs.rs`: the hardcoded snapshot of
//! every typedef name transitively exported by the build machine's real
//! `/usr/include/X11/Xlib.h` (and everything *it* includes -- X11/X.h,
//! sys/types.h, stdint.h, ...).
//!
//! Step 6b (`imports.rs`) resolves `#include <...>` against the real system
//! headers when they're present, but falls back to this hardcoded list on a
//! machine without X11 dev headers installed, so `i_video.c` parses either
//! way. Re-run this whenever the generated list should be refreshed (e.g.
//! after a distro upgrade changes Xlib.h):
//!
//! ```sh
//! cargo run --example update_xlib_typedefs
//! ```

use std::path::Path;

fn main() {
    let xlib_path = Path::new("/usr/include/X11/Xlib.h");
    if !xlib_path.is_file() {
        eprintln!(
            "error: {} not found -- install X11 dev headers (e.g. libx11-dev) on this machine before regenerating.",
            xlib_path.display()
        );
        std::process::exit(1);
    }

    let mut resolver = transpiler::parser::ImportResolver::new();
    let mut names: Vec<String> = resolver.resolve(xlib_path).into_iter().collect();
    names.sort();

    let mut out = String::new();
    out.push_str("//! Every typedef name transitively exported by `/usr/include/X11/Xlib.h`\n");
    out.push_str("//! (and everything it includes) on the machine this was generated on.\n");
    out.push_str("//!\n");
    out.push_str("//! Used by `imports.rs` as a fallback for resolving `i_video.c`'s X11 types\n");
    out.push_str("//! when real X11 dev headers aren't installed on the machine running the\n");
    out.push_str("//! pipeline. Regenerate with `cargo run --example update_xlib_typedefs`.\n");
    out.push_str("//!\n");
    out.push_str("//! GENERATED FILE -- do not hand-edit.\n\n");
    out.push_str("pub const XLIB_TYPEDEFS: &[&str] = &[\n");
    for name in &names {
        out.push_str(&format!("    \"{name}\",\n"));
    }
    out.push_str("];\n");

    let dest = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/parser/xlib_typedefs.rs");
    std::fs::write(&dest, out).expect("failed to write xlib_typedefs.rs");
    println!("Wrote {} typedef names to {}", names.len(), dest.display());
}
