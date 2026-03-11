use crate::template::FunctionTemplateInstantiation;
use quote::format_ident;
use syn::{GenericParam, ItemFn, Token, fold::Fold, parse_macro_input, punctuated::Punctuated};

mod template;

#[proc_macro_attribute]
/// This macro makes a test into a more C++ style template. The test can
/// have ONE generic parameter and we can then give different instantiations
/// for this generic parameter. It's similar in spirit to the
/// [`typed_test_gen`](https://crates.io/crates/typed_test_gen) crate,
/// but while that crate enforces the generic bounds, this one does not
/// and does a C++ style replacement of the generic argument. In general
/// the `typed_test_gen` crate is better, because it helps us enforce
/// the generic bounds and code correctness. But for some crates it's just
/// incredibly hard to talk about the generic bounds of something in the
/// test and we just want it to work...
pub fn test_template(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let concrete_types =
        parse_macro_input!(attr with Punctuated::<syn::Ident, Token![,]>::parse_terminated);
    let mut function = parse_macro_input!(item as syn::ItemFn);

    if concrete_types.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "give one or more type names in brackets",
        )
        .to_compile_error()
        .into();
    }

    let concrete_types: Vec<_> = concrete_types.into_iter().collect();

    // first we make sure that the function has exactly one generic type param

    let mut type_params = function.sig.generics.type_params();
    let Some(type_param) = type_params.next() else {
        return syn::Error::new_spanned(
            function.sig,
            "The test function must have exactly one generic type paramter",
        )
        .to_compile_error()
        .into();
    };

    if type_params.next().is_some() {
        return syn::Error::new_spanned(
            function.sig.generics,
            "The test function must have exactly one generic type paramter",
        )
        .to_compile_error()
        .into();
    }

    let template_ident = type_param.ident.clone();

    // we just remove this one ident. This leaves const generics and lifetimes
    // in there, so the compiler can complain about them once we generate the
    // test function from this. Hacky, but I don't care right now.
    remove_type_generic(&mut function, template_ident.clone());

    // now we create the "template instantiations" of the functions for the concrete types
    let template_instantiations: Vec<_> =
        std::iter::repeat_n(function, concrete_types.len()).collect();

    let template_instantiations: Vec<_> = template_instantiations
        .into_iter()
        .zip(concrete_types)
        .map(|(mut instance, concrete)| {
            instance.sig.ident = format_ident!("{}_{}", instance.sig.ident, concrete);

            // we apply this to the block which should remove all the
            let mut block = FunctionTemplateInstantiation::new(template_ident.clone(), concrete);
            instance.block = Box::new(block.fold_block(*instance.block));
            instance
        })
        .collect();

    quote::quote! {
        #(
            #[test]
            #[allow(non_snake_case)]
            #template_instantiations
        )*
    }
    .into()
}

/// remove a generic type param from the function signature
fn remove_type_generic(item_fn: &mut ItemFn, type_generic_ident: syn::Ident) {
    item_fn.sig.generics.params = item_fn
        .sig
        .generics
        .params
        .clone() // make a mutable copy
        .into_iter()
        .filter(|param| match param {
            GenericParam::Type(tp) => tp.ident != type_generic_ident,
            _ => true, // keep lifetimes and consts
        })
        .collect();
}
