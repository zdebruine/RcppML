#!/usr/bin/env Rscript
# Non-interactive CRAN submission script
# Uploads tarball, fills Step 2 form, triggers confirmation email

built_path <- "/mnt/home/debruinz/RcppML-2/RcppML_1.0.0.tar.gz"
stopifnot(file.exists(built_path))
cat("Tarball:", built_path, "\n")
cat("Size:", round(file.size(built_path) / 1e6, 1), "MB\n")

comment <- paste(readLines("/mnt/home/debruinz/RcppML-2/cran-comments.md"), collapse = "\n")

# Step 1: Upload the tarball
cat("\n=== Step 1: Uploading tarball ===\n")
h1 <- curl::new_handle()
curl::handle_setopt(h1, cookiejar = "", cookiefile = "",
                    timeout = 300, connecttimeout = 60,
                    low_speed_time = 300, low_speed_limit = 100)
curl::handle_setform(h1,
  pkg_id = "",
  name = "Zachary DeBruine",
  email = "zacharydebruine@gmail.com",
  uploaded_file = curl::form_file(built_path, type = "application/x-gzip"),
  upload = "Upload the package"
)
resp1 <- curl::curl_fetch_memory("https://xmpalantir.wu.ac.at/cransubmit/index2.php", handle = h1)
page1 <- rawToChar(resp1$content)
cat("HTTP status:", resp1$status_code, "\n")

# Extract pkg_id from the step 2 form
pkg_id_match <- regmatches(page1, regexpr('name="pkg_id"[^>]*value="[^"]+"', page1))
if (length(pkg_id_match) == 0) {
  cat("ERROR: Could not find pkg_id in response. Dumping page:\n")
  cat(page1)
  stop("Upload failed - no pkg_id found")
}
pkg_id <- sub('.*value="', '', sub('"$', '', pkg_id_match))
cat("pkg_id:", pkg_id, "\n")

# Extract all hidden inputs and form fields
inputs <- regmatches(page1, gregexpr("<input[^>]+>", page1))[[1]]
cat("\nForm inputs found:\n")
for (inp in inputs) cat("  ", inp, "\n")

# Step 2: Submit the form with comment
cat("\n=== Step 2: Submitting to CRAN ===\n")
h2 <- curl::new_handle()
curl::handle_setopt(h2, cookiejar = "", cookiefile = "", timeout = 120)
# Copy cookies from h1
cookies <- curl::handle_cookies(h1)
cat("Cookies:", nrow(cookies), "\n")

curl::handle_setform(h2,
  pkg_id = pkg_id,
  name = "Zachary DeBruine",
  email = "zacharydebruine@gmail.com",
  comment = comment,
  submit = "Submit package"
)
resp2 <- curl::curl_fetch_memory("https://xmpalantir.wu.ac.at/cransubmit/index2.php", handle = h2)
page2 <- rawToChar(resp2$content)
cat("HTTP status:", resp2$status_code, "\n")

# Check result
if (grepl("confirm", page2, ignore.case = TRUE)) {
  cat("\n=== SUCCESS: Package submitted! ===\n")
  cat("A confirmation email has been sent to zacharydebruine@gmail.com\n")
  cat("You MUST click the link in that email to complete the submission.\n")
} else {
  cat("\nResponse page content:\n")
  # Strip HTML tags for readability
  lines <- strsplit(page2, "<[^>]+>")[[1]]
  lines <- trimws(lines)
  lines <- lines[nchar(lines) > 3]
  cat(lines, sep = "\n")
}
