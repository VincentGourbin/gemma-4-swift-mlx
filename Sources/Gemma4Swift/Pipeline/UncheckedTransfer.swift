// Boite de transfert pour franchir une frontiere de tache avec un type
// non-Sendable (CGImage, MLXArray).
//
// Les preprocesseurs exposent des variantes `async` qui deportent le travail
// CPU sur une tache detachee (cf. [[Gemma4ImageProcessor]]). La valeur n'est
// jamais *partagee* : elle est construite d'un cote de la frontiere, consommee
// de l'autre, et l'original n'est plus touche. C'est un transfert, pas un
// partage — d'ou le `@unchecked`.
struct UncheckedTransfer<Value>: @unchecked Sendable {
    let value: Value

    init(_ value: Value) {
        self.value = value
    }
}
