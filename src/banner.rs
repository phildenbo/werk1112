use std::io::{self, Write};

pub const BANNER: &str = concat!(
    "\n",
    "\x1b[38;2;0;220;255m   __      __\x1b[38;2;59;130;246m ______ \x1b[38;2;99;102;241m ____  \x1b[38;2;139;92;246m __ __ \x1b[38;2;255;79;195m ___  ___  ___  _____\x1b[0m\n",
    "\x1b[38;2;0;220;255m   \\ \\    / /\x1b[38;2;59;130;246m / __/  \x1b[38;2;99;102;241m/ __ \\ \x1b[38;2;139;92;246m/ //_/\x1b[38;2;255;79;195m<  / <  / <  / |__  /\x1b[0m\n",
    "\x1b[38;2;0;220;255m    \\ \\/\\/ / \x1b[38;2;59;130;246m/ _/   \x1b[38;2;99;102;241m/ /_/ /\x1b[38;2;139;92;246m/ ,<   \x1b[38;2;255;79;195m/ /  / /  / /   /_ < \x1b[0m\n",
    "\x1b[38;2;0;220;255m     \\ /\\ /  \x1b[38;2;59;130;246m/ /___  \x1b[38;2;99;102;241m/ _, _/\x1b[38;2;139;92;246m/ /| | \x1b[38;2;255;79;195m/ /  / /  / /  ___/ / \x1b[0m\n",
    "\x1b[38;2;0;220;255m      \\__/   \x1b[38;2;59;130;246m/____/  \x1b[38;2;99;102;241m/_/ |_|\x1b[38;2;139;92;246m/_/ |_| \x1b[38;2;255;79;195m/_/  /_/  /_/  /____/  \x1b[0m\n",
    "\n",
    "\x1b[38;2;0;220;255m                Any \x1b[38;2;59;130;246mModel\x1b[38;2;99;102;241m. \x1b[38;2;139;92;246mAny\x1b[38;2;255;79;195mwhere.\x1b[0m\n",
    "\n",
);

pub fn print_banner() {
    let mut stdout = io::stdout().lock();
    let _ = stdout.write_all(BANNER.as_bytes());
    let _ = stdout.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banner_contains_required_identity() {
        let plain = strip_ansi(BANNER);
        assert!(plain.contains("Any Model. Anywhere."));
        assert!(BANNER.contains("\x1b[38;2;0;220;255m"));
        assert!(BANNER.contains("\x1b[38;2;255;79;195m"));
    }

    fn strip_ansi(text: &str) -> String {
        let mut plain = String::new();
        let mut chars = text.chars();

        while let Some(ch) = chars.next() {
            if ch == '\x1b' {
                if chars.next() == Some('[') {
                    for ch in chars.by_ref() {
                        if ch == 'm' {
                            break;
                        }
                    }
                }
                continue;
            }
            plain.push(ch);
        }

        plain
    }
}
