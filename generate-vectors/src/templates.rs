use std::{fs, path::Path};

use anyhow::Context;
use handlebars::Handlebars;

pub const ID_FIELD_NAME: &str = "__ID_FIELD__";

pub fn read_templates_from_dir<'a, P: AsRef<Path>>(
    template_dir: P,
) -> Result<(Vec<String>, Handlebars<'a>), anyhow::Error> {
    let mut template_names = vec![ID_FIELD_NAME.to_string()];
    let mut templates = Handlebars::new();
    for entry in
        fs::read_dir(template_dir).context("could not read template dir")?
    {
        let entry = entry.context("could not read entry in template dir")?;
        if !entry
            .file_type()
            .context("could not get file type")?
            .is_file()
        {
            continue;
        }
        let path = entry.path().to_owned();
        let file_name = path
            .file_name()
            .context("could not convert template file name into utf8")?
            .to_str()
            .context("could not parse file name as utf8")?;

        if !file_name.ends_with(".handlebars") {
            continue;
        }

        let template_name = path
            .file_stem()
            .expect(".handlebars but still no stem")
            .to_str()
            .expect("non utf8 file stem");

        let template_source = fs::read_to_string(entry.path())
            .context("could not read handlebars template")?;
        templates
            .register_template_string(template_name, &template_source)
            .context("could not compile template")?;
        template_names.push(template_name.to_string());
    }
    Ok((template_names, templates))
}
