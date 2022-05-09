* Change behaviour of gas basket aggregation functions to be the same as
  other aggregation functions.
  Now, `ds.pr.gas_basket_contents_sum()` and
  `ds.pr.fill_na_gas_basket_from_contents()` work like `ds.pr.sum()`.
  Both now have a `skipna` argument, and operate as if `skipna=True`
  by default.
  Note that this is a breaking change, by default NaNs are now
  treated as zero values when aggregating gas baskets!
