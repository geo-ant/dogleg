fn main() {
    #[cfg(feature = "debugging")]
    {
        // this is a bit hacky and will probably only work on my machine (TM),
        // but it allows me to enable glog (google logging library), because that
        // needs an initialization call just like the rust logging crates, but
        // there is no
        println!("cargo:rerun-if-changed=cpp/glogging.cpp.cc");
        println!("cargo:rustc-link-lib=glog");

        cc::Build::new()
            .file("cpp/glogging.cpp")
            .cpp_link_stdlib("stdc++")
            .compile("glogging_glue");
    }
}
