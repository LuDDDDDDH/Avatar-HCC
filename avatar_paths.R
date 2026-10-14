.avatar_cache <- new.env(parent = emptyenv())
.avatar_source <- local({
  location <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (is.null(location)) normalizePath(getwd(), winslash = "/", mustWork = FALSE) else normalizePath(dirname(location), winslash = "/", mustWork = FALSE)
})

.avatar_config <- function() {
  location <- Sys.getenv("AVATAR_INPUT_CONFIG", unset = "")
  if (!nzchar(location)) return(list())
  state <- file.info(location)
  key <- paste(location, state$mtime, state$size, sep = "|")
  if (exists("key", envir = .avatar_cache, inherits = FALSE) && identical(.avatar_cache$key, key)) return(.avatar_cache$value)
  value <- jsonlite::fromJSON(location, simplifyVector = FALSE)
  .avatar_cache$key <- key
  .avatar_cache$value <- value
  value
}

.avatar_get <- function(section, key, default) {
  config <- .avatar_config()
  values <- config[[section]]
  if (key %in% names(values)) return(values[[key]])
  if (missing(default)) stop(paste("Missing configured input:", key))
  default
}

get_value <- function(key, default) {
  if (missing(default)) .avatar_get("values", key) else .avatar_get("values", key, default)
}
get_column <- function(key, default) {
  value <- if (missing(default)) .avatar_get("columns", key) else .avatar_get("columns", key, default)
  if (!is.character(value) || length(value) != 1L || !nzchar(value)) stop(paste("Invalid configured column:", key))
  value
}
get_label <- function(key, default) {
  if (missing(default)) .avatar_get("labels", key) else .avatar_get("labels", key, default)
}

.avatar_root <- function() {
  value <- Sys.getenv("AVATAR_PROJECT_DIR", unset = "")
  if (!nzchar(value)) value <- Sys.getenv("AVATAR_INPUT_ROOT", unset = "")
  if (!nzchar(value)) stop("Set AVATAR_PROJECT_DIR or AVATAR_INPUT_ROOT")
  normalizePath(value, winslash = "/", mustWork = FALSE)
}

get_path <- function(key, default, root = NULL) {
  value <- if (missing(default)) .avatar_get("paths", key) else .avatar_get("paths", key, default)
  if (grepl("^(/|[[:alpha:]]:[/\\\\])", value)) return(value)
  if (is.null(root)) root <- .avatar_root()
  file.path(root, value)
}

project_path <- function(relative = "") {
  root <- .avatar_root()
  source <- .avatar_source
  if (identical(tolower(root), tolower(source)) || startsWith(tolower(source), paste0(tolower(root), "/")) || startsWith(tolower(root), paste0(tolower(source), "/"))) stop("Input root must be separate from the code directory")
  value <- get_path(relative, relative, root)
  if (grepl("(^|[/\\\\])\\.\\.([/\\\\]|$)", relative) || grepl("(^|[/\\\\])\\.\\.([/\\\\]|$)", value)) stop("Invalid input path")
  resolved <- normalizePath(value, winslash = "/", mustWork = FALSE)
  if (!identical(tolower(root), tolower(resolved)) && !startsWith(tolower(resolved), paste0(tolower(root), "/"))) stop("Input path escapes the configured root")
  value
}
