#!/usr/bin/env wolframscript
(* Regenerate the Mathematica reference values in tests/data/outputs/ from the
   notebook that defines the model (mathematica/sbm_mcmc.nb in the paper's
   reproduction repository, community-hierarchy_reproduction).

     wolframscript -file tests/mathematica/generate_fixtures.wls [path/to/sbm_mcmc.nb]

   The notebook's function definitions (every f[...] := ... in an Input cell) are
   loaded, then the cells tagged "ch-fixtures" (section "community-hierarchy test
   fixtures") are evaluated in order with fixtureDir = tests/data/outputs.
   Only the notebook is read; nothing is written to it. *)

repo = ParentDirectory[ParentDirectory[DirectoryName[$InputFileName]]];
nbPath = If[Length[$ScriptCommandLine] >= 2, $ScriptCommandLine[[2]],
   FileNameJoin[{ParentDirectory[ParentDirectory[repo]], "package-testing",
     "community-hierarchy_reproduction", "mathematica", "sbm_mcmc.nb"}]];
If[! FileExistsQ[nbPath], Print["notebook not found: ", nbPath]; Exit[1]];

(* read a copy: Get can fail on a notebook that is open in the front end *)
tmp = FileNameJoin[{$TemporaryDirectory, "ch_fixture_source.nb"}];
CopyFile[nbPath, tmp, OverwriteTarget -> True];
nb = Get[tmp];
DeleteFile[tmp];

held[b_] := Quiet[ToExpression[b, StandardForm, HoldComplete]];
stmts[HoldComplete[CompoundExpression[a___]]] := List @@ (HoldComplete /@ Unevaluated[{a}]);
stmts[h_] := {h};
isDef[HoldComplete[SetDelayed[_[___], _]]] := True;
isDef[_] := False;

inputs = Cases[nb, Cell[BoxData[b_], "Input", opts___] :> {b, CellTags /. {opts} /. CellTags -> {}}, Infinity];
tagged[t_] := MemberQ[Flatten[{t}], "ch-fixtures"];
Scan[ReleaseHold, Select[Flatten[stmts[held[#[[1]]]] & /@ Select[inputs, ! tagged[#[[2]]] &]], isDef]];

fixtureDir = FileNameJoin[{repo, "tests", "data", "outputs"}];
fixtureCells = Select[inputs, tagged[#[[2]]] &];
If[fixtureCells === {}, Print["no cells tagged ch-fixtures in ", nbPath]; Exit[1]];
Scan[ReleaseHold[held[#[[1]]]] &, fixtureCells];
Print["evaluated ", Length[fixtureCells], " fixture cells from ", nbPath, " -> ", fixtureDir];
