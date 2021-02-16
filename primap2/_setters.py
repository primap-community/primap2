import typing

import numpy as np
import xarray as xr

from . import _accessor_base


class DataArraySettersAccessor(_accessor_base.BaseDataArrayAccessor):
    def set(
        self,
        dim: str,
        key: typing.Any,
        value: typing.Union[xr.DataArray, np.ndarray],
        *,
        value_dims: typing.Optional[typing.List[str]] = None,
        existing: str = "error",
    ) -> xr.DataArray:
        """Set values, expanding the given dimension as necessary.

        The handling of already existing key values can be selected using the
        ``existing`` parameter.

        Parameters
        ----------
        dim: str
            Dimension along which values should be set.
        key: scalar or list of scalars
            Keys in the dimension which should be set. Key values which are missing
            in the dimension are inserted. The handling of key values which already
            exist in the dimension is determined by the ``existing`` parameter.
        value: xr.DataArray or np.ndarray which can be broadcast to ``da[{dim: key}]``
            Values that will be inserted at the positions specified by ``key``.
        value_dims: list of str, optional
            Specifies the dimensions of ``value``. If ``value`` is not a DataArray
            and ``da[{dim: key}]`` is higher-dimensional, it is necessary to specify
            the value dimensions.
        existing: "error", "overwrite", or "fillna", optional
            How to handle existing keys. If ``existing="error"`` (default), a ValueError
            is raised if any key already exists. If ``existing="overwrite"``, new values
            overwrite current values for existing keys. If ``existing="fillna"``, the
            new values only overwrite NaN values for existing keys.

        Returns
        -------
        da : xr.DataArray
            modified DataArray
        """
        dim = self._da.pr.dim_alias_translations.get(dim, dim)
        if dim not in self._da.dims:
            raise ValueError(f"Dimension {dim!r} does not exist.")

        if value_dims is not None:
            value_dims = [
                self._da.pr.dim_alias_translations.get(x, x) for x in value_dims
            ]

        if np.ndim(key) == 0:  # scalar
            key = [key]
        else:
            key = key

        if existing == "error":
            already_existing = set(self._da[dim].values).intersection(set(key))
            if already_existing:
                raise ValueError(
                    f"Values {already_existing!r} for {dim!r} already exist."
                    " Use existing='overwrite' or 'fillna' to avoid this error."
                )
            # without conflicting keys, the new keys are overwritten
            existing = "fillna"

        if isinstance(value, xr.DataArray):
            if value_dims is not None:
                raise ValueError("value_dims given, but value is already a DataArray.")

            # conform value to given dim: key
            if dim not in value.dims:
                if dim in value.coords:
                    value = value.reset_coords(dim, drop=True)
                value = value.expand_dims({dim: key})
            else:
                value = value.loc[{dim: key}]
            value, expanded = xr.broadcast(value, self._da)
            value.attrs = self._da.attrs
            value.name = self._da.name

        else:
            new_index = list(self._da[dim].values)
            for item in key:
                if item not in new_index:
                    new_index.append(item)
            new_index = np.array(new_index, dtype=self._da[dim].dtype)
            expanded = self._da.reindex({dim: new_index}, copy=False)
            sel = expanded.loc[{dim: key}]

            if value_dims is None:
                value_dims = []
                for i, dim in enumerate(sel.dims):
                    if sel.shape[i] > 1:
                        value_dims.append(dim)
                if len(value_dims) != len(value.shape):
                    raise ValueError(
                        "Could not automatically determine value dimensions, please"
                        " use the value_dims parameter."
                    )
            value = xr.DataArray(
                value,
                coords=[(dim, sel[dim]) for dim in value_dims],
                name=self._da.name,
                attrs=self._da.attrs,
            ).broadcast_like(sel)

        if existing == "overwrite":
            return value.combine_first(expanded)
        elif existing == "fillna":
            return expanded.combine_first(value)
        else:
            raise ValueError(
                "If given, 'existing' must specify one of 'error', 'overwrite', or"
                f" 'fillna', not {existing!r}."
            )
