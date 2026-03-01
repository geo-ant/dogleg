use syn::{Ident, ItemFn};

pub struct FunctionTemplateInstantiation {
    template_ident: Ident,
    concrete_type: Ident,
}

impl FunctionTemplateInstantiation {
    pub fn new(template_ident: Ident, concrete_type: syn::Ident) -> Self {
        Self {
            template_ident,
            concrete_type,
        }
    }
}

impl syn::fold::Fold for FunctionTemplateInstantiation {
    fn fold_path(&mut self, mut path: syn::Path) -> syn::Path {
        if let Some(segment) = path.segments.first_mut() {
            if segment.ident == self.template_ident {
                segment.ident = self.concrete_type.clone();
            }
        }
        path
    }
}
