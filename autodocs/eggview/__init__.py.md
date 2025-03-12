## function `write_page`

Writes an HTML page by replacing template placeholders with provided content.

**Arguments:**

* `fout`: File object to write the HTML page to.
* `main_content`: String containing the main content to be inserted into the template.

**Functionality:**

* Reads template and JavaScript files from specified file paths.
* Replaces template placeholders with the provided `main_content`, JavaScript code, and CSS styles.
* Writes the modified template to the specified `fout` file.
