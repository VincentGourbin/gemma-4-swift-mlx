// Boite de transfert pour faire franchir une frontiere de tache a un MLXArray.
//
// `MLXArray` n'est pas `Sendable` (et n'est pas thread-safe : cf. la doc
// mlx-swift). Les surcharges `async` des processeurs d'image ont malgre tout
// besoin de rapatrier un resultat depuis leur tache detachee.
//
// Ce que cette boite suppose, et qui doit rester vrai a chaque site d'appel :
// la valeur est **materialisee** (`eval`) avant d'etre emballee, et l'exemplaire
// cote tache n'est plus reference apres. C'est un transfert de propriete, pas un
// partage — sans l'`eval` prealable ce serait un graphe paresseux construit ici
// et evalue ailleurs, ce que mlx-swift interdit explicitement.
//
// A ne pas etendre a d'autres types sans revalider cette precondition. Notamment
// `CGImage` est deja `Sendable` et n'a rien a faire ici.
struct UncheckedTransfer<Value>: @unchecked Sendable {
    let value: Value

    init(_ value: Value) {
        self.value = value
    }
}
