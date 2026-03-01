use quote::format_ident;
use syn::{GenericParam, ItemFn, Token, fold::Fold, parse_macro_input, punctuated::Punctuated};

use crate::template::FunctionTemplateInstantiation;

mod template;

#[proc_macro_attribute]
/// The principal macro attribute in this crate that lets us keep function
/// documentation in sync with the function signature.
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
            function.sig.generics,
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
