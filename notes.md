# What to do with sparse data?


## Background / Motivation

* Our PRIMAP-hist datasets have a lot of NaNs, which makes tasks memory-intensive
* There are different approaches to fix this problem.
* We already use parts of one approach by merging "source" and "scenario" into one dimension.
* For us, it would be better if we could still access these two dimensions.
* An example: In the CTFs each country has its own animals, some animals only exist in certain countries. For example, llamas only exist in three countries
* All other countries will also have the dimension "llama", even though all values are NAN.
* We also want to be able to access data from different countries and sources without looking in different repositories. For example, we get a list of 10 countries and want to find all the relevant data.
* From sparse documentation: Sparse arrays, or arrays that are mostly empty or filled with zeros, are common in many scientific applications. To save space we often avoid storing these arrays in traditional dense formats, and instead choose different data structures. Our choice of data structure can significantly affect our storage and computational costs when working with these arrays.


## Our specific requirements

We are completely sure what our requirements are exactly. We have some ideas, but we need to discuss them further. Here are some of them:

* **Merge datasets**
    * Example: This is a script that likely blows up on our laptops. It compiles all CRF data sets for one submission year into one data set. `src/unfccc_ghg_data/unfccc_crf_reader/crf_raw_for_year.py`. The data sets are sparse and need a lot of memory. We want to reduce memory significantly.
* **Select**
  * Filtering data sets is another essential task. We want to be able to filter data sets by different dimensions, e.g. country, sector, gas, etc. Our primap2 loc function already does this.
* **Set**
  * The same goes for assigning values to a data set. We want to be able to set values for different dimensions.
* **Downscale**
  * We want to be able to downscale data sets.
* We want to keep as much as possible of our current code base. But we will likely have to rewrite some parts of it.


## Approaches

* There are algorithms with which we can process data sets more efficiently
* We can rebuild the data set structures. That could be within xarray or even beyond that.
* We can also set up a separate database.


### SQL data base


* Imagine a long format table in a SQL table
* We could host a backend or use sqlite
* It could be one big file or several small files
* We (Mika) have a pretty good idea how to implement this and what the technology stack would look like.

<table>
  <tr>
   <td>pro
   </td>
   <td>con
   </td>
  </tr>
  <tr>
   <td>
<ul>

<li>can handle large amounts of data</li>
<li>we know the technology</li>
</ul>
   </td>
   <td>
<ul>

<li>needs redesign of every step in pipeline</li>
<li>not version controlled</li>
</ul>
   </td>
  </tr>
</table>

### Xarray data tree

* data trees are a hierarchical structure of data sets
* We would not have to merge data sets to perform filter / select operations for all available data
* Would we be able to downscale as well?
* Would it be an option to use data tree only for pre-processing?
* Can we throw in data tree and keep most of our set up? Probably not, but it may not be super difficult either
* Suitable for storage and query
* https://www.youtube.com/watch?v=Iwcuy6Smbjs
* DataTree is a hierarchical tree of datasets - it replaces looping through datasets to perform an operation
* They talk a lot about optimising for cloud use.
* DataTree was a separate repository for some years and is now part of xarray. So it should not be experimental anymore.
* At the same time the documentation does not feel comprehensive.
* In terms of our sparsity issue: if we combine several datasets in one datatree it would help to some extent, but would not eliminate the problem. For example, a unique category in data set A would not create all NaNs in for that dimension in data set B, but still create a lot of NaNs in data set A
* Out of 6 devs, half are from NASA’s climate division. Do they still have jobs?
* It’s relatively easy to convert several data sets into a xarray data tree
* What is the benefit of having data tree is we don’t perform operations on the data tree (which would be mapped over the sub-trees)?
* In the example crf_raw_for_year.py in UNFCCC_non-AnnexI_data:
    * We can build a data tree of all sub-data sets, but it is not possible to save in IF. We could write a function that converts from data tree to interchange formats. That would require to merge the data sets either in the traditional way - pr-merge -  (that’s what we’re trying to avoid) or we first convert to IF and then merge (pd.concatatenate).
    * Data tree has a .to_netcfd() method. If we save all data sets as data tree and only convert / merge when needed, it could save a processing time
* Is it possible to filter by a data variable, e.g. “CO2” for all data sets in a data tree?

<table>
  <tr>
   <td>pro
   </td>
   <td>con
   </td>
  </tr>
  <tr>
   <td>
<ul>

<li>Supports conversion from/to netcfd and zarr</li>
<li>We can stay in the xarray world</li>
<li>We could use it to process and save data sets</li>
</ul>
   </td>
   <td>
<ul>

<li>Is a rather new feature of xarray</li>
<li>We need to make our primap2 functions compatible with data tree</li>
<li>Processing data trees may involve looping through the sub-datasets, is it faster than merging once?</li>
<li>The hierarchical concept does not really align with our use-case. We would have all country data sets as "siblings". But wouldn't be a problem either</li>
</ul>
   </td>
  </tr>
</table>


###


### sparse arrays


#### How does it work?

* In primap2 we could write a function that turns a data set’s array into a sparse array
* We already use duck arrays in primap2 - pint arrays
* Not all numpy array methods are implemented for sparse arrays.
* duck arrays with sparse reduce memory significantly. A typical PRIMAP-hist data set has a coverage (non-nan / all values) from 21% for CO2 and 1% for NF3
* converting from numpy arrays to sparse array and back is easy
* Our merge function in primap2 works with some minor tweaks for sparse arrays
* The set function is harder to implement with sparse arrays
* Our problem is: sparse and numpy arrays may be easy to swap, but with xarray as an extra layer things get more complicated. It would be possible to move all operations to numpy array level. That also means we need to handle indexing and units (in pint) explicitly

Some more considerations for duck arrays:

* Can we use pr.set with sparse?
    * This function has different ways to handle existing keys ("fillna_empty", "error", "overwrite", or "fillna"), they all need to be considered
    * We would likely have to move operations from xarray to numpy, e.g. from xr.where() to np.where(). It's possible but could be tricky when dealing with dimensions
    * pr.set allows to overwrite existing values only if all existing values are NaN - we need to check if this is supported by sparse arrays
    * When using xarray operations it seems to get confused with pint units sometimes
* Converting to sparse arrays only works for numpy arrays, not for xarray data sets -> we have to loop through all data sets and convert them to numpy arrays


<table>
  <tr>
   <td>pro
   </td>
   <td>con
   </td>
  </tr>
  <tr>
   <td>
<ul>

<li>It is designed to solve our problem</li>
<li>easy to convert back and forth</li>
<li>It is designed to solve our problem</li>
</ul>
   </td>
   <td>
<ul>
<li>Does not support all xarray functions</li>
<li>It is not an easy replacement. We would have to go to every primap2 function and ask, would this work with sparse arrays?</li>
<li>Requires careful handling of indexing and units</li>
</ul>
   </td>
  </tr>
</table>



###


### Pandas



* We already use pandas data frames in primap2 - the interchange format (IF)
* Some operations would be straightforward in pandas, for example a merge means simply stacking to tables. We already use this approach in data reading.
* PRIMAP-hist as a wide data frame would make mathematical operations very slow
* Xarray is better with unit support (we use pint arrays), the unit support in pandas is not ideal, but we can add extra columns at little costs
* Using pandas would help us in some case. When merging datasets, pandas is quicker than other approaches, because we can simply stack tables. However, it is not equivalent to our primap2 merge function because data consistency is not checked. A function to combine interchange format datasets does not exist yet. When converting back to xarray, the data set would still be sparse and therefore difficult to handle.


### Proposed solution

#### Option 1: Data tree for pre-processing
- Everything before the CSG is done with data tree
- We save the data tree as netcfd
- In the CSG we merge to a single data set. With only the primap-hist sectors, the sparsity is manageable

<table>
  <tr>
   <td>pro
   </td>
   <td>con
   </td>
  </tr>
  <tr>
   <td>
<ul>

<li>No additional dependencies, it's all xarray</li>
</ul>
   </td>
   <td>
<ul>
<li>Every function that deals with data sets needs to be adapted to data tree objects</li>
<li>We add another data type to the primap2 ecosystem</li>
<li>Looping through data sets may be necessary for some tasks</li>
</ul>
   </td>
  </tr>
</table>


Other notes:

- wie misst man memory gut? große Daten nehmen und schauen wie viel es im Betriebssystem in Anspruch nimmt,RES resident, wenn es primap-hist

- top anschauen, memory usage herausfinden

- Für was wäre SQL eine Antwort

- syntethischer Testdatensatz der explodiert weil er zu sparse ist

Beispiele:

UNFCCC nonAnnexI compile CRF data, Testdatensatz rausziehen

CRT Datensatz was Johannes gesagt hat

- großen Datensatz laden, mehrere GB, dann kann man python prozess vernachlässigen

- zarr dateien sind cloud ready

- use case: Ich will alle Daten haben, eher als subsets und joins wie man si in SQL macht

- HDF5 Gruppen (DataTree) oder filenamen convention, aufbauend auf lokalen Daten

- Wir würden nicht unbedingt alles in eine zentrale Datenbank legen

- use cases:

	- Kann ich mit den eingelesenen Daten, sowas machen wie downscaling (muss nicht genau die Funktion sein) oder GHG Konvertierung (primap2 Funktion ) Muss ich einen loop schreiben?

	- Kann ich alle meine Daten speichern und daraus dann einzelne Teile rausziehen, hier könnte man über Datenbank systeme nachdenken, kann ich aus meinem großen Objekt was auxch sparse ist damit arbeiten, getten und setten reicht uns nicht, würde nur data storage use case lösen, hierfür brauchen wir auch merge

	- Kann man darin arbeiten oder nur speichern, wir brauchen eigentlich beides, wir müssen eine Ahnung haben was die beiden sachen können

	- Ergebnis: wir können folgende use cases mit data tree lösen und andere mit sparse

	- Der reine storage use case:, kann man einen DataTree machen? Mit data tree kann man auch nur die primap-hist sektoren wählen und das wäre dann nicht so sparse, wir können dieses Skript auch mal mit sparse array umschreiben

#### To Dos:
- pre prcoessing in data tree, zusammenführen vor CSG, der braucht nur primap-hist kategorien, dann geht es eh wieder bzgl memory
- Vorschläge in Berlin diskutieren
- oder nur pandas im preprocessing, dann ist es nicht mehr so sparse
- wie aufwändig ist category conversion in pandas und data tree
-
